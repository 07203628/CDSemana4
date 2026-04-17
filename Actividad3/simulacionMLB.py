import os
import math
import re
import random
import warnings
import time
from functools import lru_cache
from io import StringIO

from pybaseball import batting_stats
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry

warnings.simplefilter("ignore", InsecureRequestWarning)

columnas_interes = [
    'Name', 'Team', 'PA', 'wOBA', 'wRAA', 'Batting', 'BaseRunning',
    'Fielding', 'Positional', 'Off', 'Def', 'WAR'
]

PRESUPUESTO_INICIAL = 50_000_000
ROSTER_SLOT_ORDEN = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache_hola")
CACHE_TTL_SECONDS = 60 * 60 * 8  # 8 horas


def _crear_http_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_SESSION = _crear_http_session()


def _http_get(url, timeout=12, **kwargs):
    return HTTP_SESSION.get(url, timeout=timeout, **kwargs)


def _to_number(valor):
    numero = pd.to_numeric(valor, errors="coerce")
    if pd.isna(numero):
        return float("nan")
    return float(numero)


def _es_comando_shell(texto):
    comando = str(texto or "").strip().lower()
    if not comando:
        return False
    prefijos = (
        "source ",
        "python ",
        "/users/",
        "cd ",
        "pip ",
        "git ",
        "conda ",
        "poetry ",
    )
    return comando.startswith(prefijos)


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(nombre):
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, nombre)


def _cargar_cache_df(nombre, ttl=CACHE_TTL_SECONDS):
    path = _cache_path(nombre)
    if not os.path.exists(path):
        return None
    try:
        if (time.time() - os.path.getmtime(path)) > ttl:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def _guardar_cache_df(df, nombre):
    path = _cache_path(nombre)
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass


def normalizar_nombre(valor):
    return re.sub(r"[^a-z0-9]+", "", str(valor).lower())


def cargar_stats(season=2025):
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={season}&sportIds=1&limit=10000"
    )
    respuesta = _http_get(url, timeout=30)
    respuesta.raise_for_status()

    data = respuesta.json()
    stats = data.get("stats", [])
    if not stats:
        raise RuntimeError("La respuesta de MLB API no contiene bloque de stats.")

    splits = stats[0].get("splits", [])
    filas = []
    for split in splits:
        fila = {
            "Name": split.get("player", {}).get("fullName", ""),
            "PlayerID": split.get("player", {}).get("id"),
            "Team": split.get("team", {}).get("name", ""),
            "League": split.get("league", {}).get("name", ""),
            "Pos": split.get("position", {}).get("abbreviation", ""),
        }
        fila.update(split.get("stat", {}))
        filas.append(fila)

    return pd.DataFrame(filas)


def cargar_equipos_mlb(season=2025):
    cache = _cargar_cache_df(f"equipos_{season}.csv")
    if cache is not None and not cache.empty:
        return cache

    url = f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}"
    respuesta = _http_get(url, timeout=12)
    respuesta.raise_for_status()

    equipos = respuesta.json().get("teams", [])
    filas = []
    for equipo in equipos:
        league = equipo.get("league", {}) or {}
        division = equipo.get("division", {}) or {}
        filas.append({
            "TeamID": equipo.get("id"),
            "Team": equipo.get("name", ""),
            "TeamAbbr": equipo.get("abbreviation", ""),
            "TeamLeague": league.get("name", ""),
            "TeamDivision": division.get("name", ""),
        })

    df = pd.DataFrame(filas)
    _guardar_cache_df(df, f"equipos_{season}.csv")
    return df


def cargar_standings_mlb(season=2025):
    cache = _cargar_cache_df(f"standings_{season}.csv")
    if cache is not None and not cache.empty:
        return cache

    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
    respuesta = _http_get(url, timeout=12)
    respuesta.raise_for_status()

    records = respuesta.json().get("records", [])
    filas = []
    for record in records:
        league = record.get("league", {}) or {}
        division = record.get("division", {}) or {}
        for team_record in record.get("teamRecords", []):
            team = team_record.get("team", {}) or {}
            league_record = team_record.get("leagueRecord", {}) or {}
            filas.append({
                "TeamID": team.get("id"),
                "Team": team.get("name", ""),
                "Wins": int(team_record.get("wins", 0) or 0),
                "Losses": int(team_record.get("losses", 0) or 0),
                "League": league.get("name", ""),
                "Division": division.get("name", ""),
                "LeagueRank": int(team_record.get("leagueRank", 0) or 0),
                "DivisionRank": int(team_record.get("divisionRank", 0) or 0),
                "Pct": league_record.get("pct", ""),
            })

    df = pd.DataFrame(filas)
    _guardar_cache_df(df, f"standings_{season}.csv")
    return df


def calcular_presupuesto_inicial(wins):
    wins = int(wins or 0)
    return 30_000_000 + max(0, 162 - wins) * 1_000_000


def _normalizar_equipo_texto(valor):
    return normalizar_nombre(valor)


def _coincide_equipo(jugador, equipo):
    candidato = _normalizar_equipo_texto(jugador.get("Team", ""))
    objetivo = _normalizar_equipo_texto(equipo)
    return objetivo in candidato or candidato in objetivo


def seleccionar_equipo_franchise(equipos_df, standings_df):
    vista = equipos_df.copy()
    if standings_df is not None and not standings_df.empty:
        vista = vista.merge(
            standings_df[["TeamID", "Wins", "Losses", "League", "Division", "LeagueRank", "DivisionRank"]],
            on="TeamID",
            how="left",
        )
    else:
        vista["Wins"] = pd.NA
        vista["Losses"] = pd.NA

    vista = vista.sort_values(by=["TeamLeague", "TeamDivision", "Team"], na_position="last").reset_index(drop=True)

    print("\n=== ELIGE TU EQUIPO ===")
    for idx, row in vista.iterrows():
        wins = row.get("Wins")
        losses = row.get("Losses")
        record = f"{int(wins)}-{int(losses)}" if pd.notna(wins) and pd.notna(losses) else "N/A"
        print(f"{idx + 1:>2}. {row.get('Team')} | {row.get('TeamLeague')} | {row.get('TeamDivision')} | {record}")

    while True:
        eleccion = input("\nNumero del equipo: ").strip()
        if _es_comando_shell(eleccion):
            print("Parece un comando de terminal. Escribe solo el numero del equipo (1-30).")
            continue
        if eleccion.isdigit() and 1 <= int(eleccion) <= len(vista):
            elegido = vista.iloc[int(eleccion) - 1].to_dict()
            return elegido
        print("Seleccion invalida.")


def construir_roster_equipo_base(stats_df, team_name, presupuesto_total):
    roster = crear_roster()
    usados = set()

    objetivo = _normalizar_equipo_texto(team_name)
    team_series = stats_df.get("Team", pd.Series("", index=stats_df.index)).astype(str).map(_normalizar_equipo_texto)
    candidatos_equipo = stats_df[(team_series.str.contains(re.escape(objetivo), na=False)) | (team_series.map(lambda x: bool(x) and x in objetivo))].copy()
    if candidatos_equipo.empty:
        roster_fallback, _ = construir_roster_optimo_con_presupuesto(stats_df, presupuesto_total, equipo_preferido=team_name)
        if roster_fallback is not None:
            return roster_fallback, 0
        return roster, 0

    candidatos_equipo["WAR"] = pd.to_numeric(candidatos_equipo.get("WAR"), errors="coerce").fillna(0.0)
    candidatos_equipo["Salary"] = pd.to_numeric(candidatos_equipo.get("Salary"), errors="coerce")
    candidatos_equipo = candidatos_equipo.sort_values(by=["WAR", "Salary"], ascending=[False, True], na_position="last")

    # Primero llena posiciones defensivas naturales para conservar mejor el roster real del equipo.
    for slot in [s for s in ROSTER_SLOT_ORDEN if s != "DH"]:
        for _, jugador in candidatos_equipo.iterrows():
            nombre = str(jugador.get("Name", "")).strip().lower()
            if not nombre or nombre in usados:
                continue
            posiciones = _slots_para_posicion(jugador.get("Pos") or jugador.get("SpotracPos") or "")
            if slot in posiciones:
                roster[slot] = jugador.to_dict()
                usados.add(nombre)
                break

    if roster["DH"] is None:
        for _, jugador in candidatos_equipo.iterrows():
            nombre = str(jugador.get("Name", "")).strip().lower()
            if nombre and nombre not in usados:
                roster["DH"] = jugador.to_dict()
                usados.add(nombre)
                break

    # Si faltan huecos, completa con una solucion optima priorizando el equipo elegido.
    faltantes = [slot for slot, jugador in roster.items() if jugador is None]
    if faltantes:
        roster_fallback, _ = construir_roster_optimo_con_presupuesto(stats_df, presupuesto_total, equipo_preferido=team_name)
        if roster_fallback is not None:
            for slot in faltantes:
                if roster[slot] is None:
                    roster[slot] = roster_fallback.get(slot)

    return roster, 0


def construir_pitching_resumen(pitchers_df):
    if pitchers_df.empty:
        return pd.DataFrame()

    bullpen = pitchers_df.copy()
    bullpen["IP"] = pd.to_numeric(bullpen["IP"], errors="coerce")
    bullpen["ERA"] = pd.to_numeric(bullpen["ERA"], errors="coerce")
    bullpen = bullpen[pd.notna(bullpen["IP"]) & pd.notna(bullpen["ERA"])].copy()
    if bullpen.empty:
        return bullpen

    liga_era = bullpen["ERA"].mean()
    grupos = []
    for team, grupo in bullpen.groupby("Team"):
        ip_total = grupo["IP"].sum()
        if ip_total <= 0:
            continue
        era_equipo = (grupo["ERA"] * grupo["IP"]).sum() / ip_total
        grupos.append({
            "Team": team,
            "PitchScore": round(liga_era - era_equipo, 2),
            "PitchERA": round(era_equipo, 2),
            "PitchIP": round(ip_total, 1),
        })

    return pd.DataFrame(grupos).sort_values(by="PitchScore", ascending=False).reset_index(drop=True)


def vender_jugador_franquicia(roster, presupuesto_restante):
    ocupados = [(slot, jugador) for slot, jugador in roster.items() if jugador is not None]
    if not ocupados:
        print("\nNo hay jugadores para vender.")
        return presupuesto_restante

    print("\n=== JUGADORES EN ROSTER ===")
    for idx, (slot, jugador) in enumerate(ocupados, start=1):
        salario = _formatear_salario(jugador.get("Salary"))
        print(f"{idx:>2}. {slot} - {jugador.get('Name')} ({jugador.get('Team')}) - {salario}")

    seleccion = input("Numero del jugador a vender: ").strip()
    if not seleccion.isdigit():
        print("\nSeleccion invalida.")
        return presupuesto_restante

    indice = int(seleccion) - 1
    if indice < 0 or indice >= len(ocupados):
        print("\nEl numero esta fuera de rango.")
        return presupuesto_restante

    slot, jugador = ocupados[indice]
    salario = _to_number(jugador.get("Salary"))
    if pd.isna(salario):
        salario = 0

    roster[slot] = None
    presupuesto_restante += int(salario)
    print(f"\nVendiste a {jugador.get('Name')} y liberaste ${int(salario):,}.")
    print(f"Presupuesto restante: ${presupuesto_restante:,.0f}")
    return presupuesto_restante


def intentar_comprar_jugador_franquicia(stats_df, roster, presupuesto_restante):
    presupuesto_restante = menu_busqueda_bateadores(stats_df, roster, "Franquicia", presupuesto_restante)
    return presupuesto_restante


def mostrar_tablero_franquicia(team_info, roster, presupuesto_restante, season_num, wins_previos=None):
    limpiar_pantalla()
    print("=== MODO FRANQUICIA ===")
    print(f"Equipo: {team_info.get('Team')}")
    print(f"Temporada simulada: {season_num}")
    if wins_previos is not None:
        print(f"Ultima marca: {wins_previos} victorias")
    print(f"Presupuesto restante: ${presupuesto_restante:,.0f}")
    mostrar_roster(roster, presupuesto_restante=presupuesto_restante, limpiar=False)


def simular_temporada_franquicia(team_name, roster, bullpen_scores, season_num):
    if any(slot is None for slot in roster.values()):
        print("\nEl roster esta incompleto.")
        return None

    offense = _fuerza_equipo_draft(roster)
    pitch_score = 0.0
    if bullpen_scores is not None and not bullpen_scores.empty and team_name in set(bullpen_scores["Team"]):
        pitch_score = float(bullpen_scores.loc[bullpen_scores["Team"] == team_name, "PitchScore"].iloc[0])

    team_strength = offense + (pitch_score * 5)
    rng = random.Random(1000 + season_num)
    wins_estimados = []
    for _, fila in bullpen_scores.iterrows():
        score = 81 + (float(fila.get("PitchScore", 0)) * 4) + rng.gauss(0, 4)
        wins_estimados.append({
            "Team": fila.get("Team", ""),
            "Wins": int(round(score)),
            "PitchScore": float(fila.get("PitchScore", 0)),
        })

    nuestro_equipo = {
        "Team": team_name,
        "Wins": int(round(81 + (team_strength - 30) + rng.gauss(0, 3))),
        "PitchScore": pitch_score,
        "Offense": round(offense, 1),
        "Strength": round(team_strength, 1),
    }

    liga = pd.DataFrame(wins_estimados)
    liga = pd.concat([liga, pd.DataFrame([{
        "Team": nuestro_equipo["Team"],
        "Wins": nuestro_equipo["Wins"],
        "PitchScore": nuestro_equipo["PitchScore"],
    }])], ignore_index=True)
    liga = liga.sort_values(by="Wins", ascending=False).reset_index(drop=True)

    posicion = liga.index[liga["Team"] == team_name].tolist()[0] + 1
    top_12 = liga.head(12)["Team"].tolist()
    campeon = False

    if team_name in top_12:
        seed_order = top_12
        strength_map = {fila["Team"]: 30 + (float(fila.get("PitchScore", 0)) * 4) for _, fila in bullpen_scores.iterrows()}
        strength_map[team_name] = team_strength
        actual = team_name
        campeon = True
        for fase, mejor_de in [("Wild Card", 3), ("Division Series", 5), ("League Championship Series", 7), ("World Series", 7)]:
            rivales = [team for team in seed_order if team != actual]
            if not rivales:
                campeon = False
                break
            rival = rivales[0]
            ganador = _simular_series(strength_map.get(actual, 30), strength_map.get(rival, 30), mejor_de=mejor_de)
            if ganador != "A":
                campeon = False
                break

    return {
        "Team": team_name,
        "Season": season_num,
        "Wins": int(nuestro_equipo["Wins"]),
        "Position": posicion,
        "Offense": round(offense, 1),
        "PitchScore": round(pitch_score, 2),
        "Strength": round(team_strength, 1),
        "Champion": campeon,
    }


def jugar_franquicia():
    try:
        print("Cargando stats de bateo...")
        stats_2025, fuente = cargar_stats_2025()
    except RuntimeError as error_carga:
        print(f"Error al cargar los datos: {error_carga}")
        return

    print("Cargando equipos...")
    equipos_mlb = cargar_equipos_mlb(2025)
    print("Cargando standings...")
    standings_2025 = cargar_standings_mlb(2025)
    print("Cargando precios de mercado...")
    precios_spotrac = cargar_precios_spotrac(2025)
    print("Preparando dataset de hitters...")
    stats_2025 = enriquecer_con_precios(stats_2025, precios_spotrac)
    stats_2025, _ = completar_stats_avanzadas_aprox(stats_2025, fuente)
    stats_2025 = enriquecer_con_equipos(stats_2025, equipos_mlb)
    print("Cargando pitcheo y resumen de liga...")
    pitchers_2025 = cargar_pitching_mlb(2025)
    pitchers_2025 = enriquecer_pitchers_con_precios(pitchers_2025, precios_spotrac)
    bullpen_scores = construir_pitching_resumen(pitchers_2025)

    equipo = seleccionar_equipo_franchise(equipos_mlb, standings_2025)
    presupuesto_restante = calcular_presupuesto_inicial(equipo.get("Wins", 81))
    roster, gastado = construir_roster_equipo_base(stats_2025, equipo.get("Team", ""), presupuesto_restante)
    presupuesto_restante = max(0, presupuesto_restante - gastado)

    if roster is None:
        roster = crear_roster()

    season_num = 2025
    campeon = False
    ultima_temporada = None

    while True:
        mostrar_tablero_franquicia(equipo, roster, presupuesto_restante, season_num, ultima_temporada["Wins"] if ultima_temporada else None)
        print("1) Vender jugador")
        print("2) Comprar jugador")
        print("3) Simular temporada")
        print("4) Ver roster")
        print("0) Salir")

        opcion = input("\nElige una opcion: ").strip()

        if opcion == "0":
            print("\nSaliendo de franquicia.")
            break
        if opcion == "1":
            presupuesto_restante = vender_jugador_franquicia(roster, presupuesto_restante)
            input("\nPresiona Enter para continuar...")
            continue
        if opcion == "2":
            presupuesto_restante = intentar_comprar_jugador_franquicia(stats_2025, roster, presupuesto_restante)
            input("\nPresiona Enter para continuar...")
            continue
        if opcion == "3":
            ultima_temporada = simular_temporada_franquicia(equipo.get("Team", ""), roster, bullpen_scores, season_num)
            if ultima_temporada is None:
                input("\nPresiona Enter para continuar...")
                continue
            print(f"\nTemporada {season_num}: {ultima_temporada['Wins']} victorias | puesto #{ultima_temporada['Position']}")
            if ultima_temporada["Champion"]:
                campeon = True
                print("\nObjetivo cumplido: llegaste a ganar la Serie Mundial.")
                input("\nPresiona Enter para salir...")
                break
            print("Aun no alcanzas el campeonato. Puedes vender y comprar antes de la siguiente temporada.")
            season_num += 1
            input("\nPresiona Enter para seguir a la siguiente temporada...")
            continue
        if opcion == "4":
            mostrar_roster(roster, stats_2025, presupuesto_restante)
            input("\nPresiona Enter para continuar...")
            continue

        print("\nOpcion invalida.")
        input("Presiona Enter para continuar...")

    if campeon:
        print("\nLa franquicia alcanzo el objetivo de roster campeon.")


@lru_cache(maxsize=4096)
def _obtener_pais_jugador_mlb(player_id):
    if player_id is None or player_id == "" or pd.isna(player_id):
        return pd.NA

    try:
        respuesta = _http_get(f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}", timeout=5)
        respuesta.raise_for_status()
        gente = respuesta.json().get("people", [])
        if not gente:
            return pd.NA
        return gente[0].get("birthCountry") or pd.NA
    except Exception:
        return pd.NA


def enriquecer_con_pais(stats_df):
    combinado = stats_df.copy()
    if "Country" in combinado.columns:
        return combinado

    if "PlayerID" not in combinado.columns:
        combinado["Country"] = pd.NA
        return combinado

    ids = (
        pd.to_numeric(combinado["PlayerID"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    id_to_country = {}
    chunk_size = 50
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        if not chunk:
            continue
        person_ids = ",".join(str(pid) for pid in chunk)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={person_ids}"
        try:
            respuesta = _http_get(url, timeout=8)
            respuesta.raise_for_status()
            people = respuesta.json().get("people", [])
            for persona in people:
                pid = persona.get("id")
                if pid is not None:
                    id_to_country[int(pid)] = persona.get("birthCountry") or pd.NA
        except Exception:
            for pid in chunk:
                id_to_country[pid] = _obtener_pais_jugador_mlb(pid)

    def _pais_para_fila(player_id):
        pid = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
        if pd.isna(pid):
            return pd.NA
        return id_to_country.get(int(pid), pd.NA)

    combinado["Country"] = combinado["PlayerID"].apply(_pais_para_fila)
    return combinado


def enriquecer_con_equipos(stats_df, equipos_df):
    if equipos_df.empty:
        combinado = stats_df.copy()
        combinado["TeamLeague"] = pd.NA
        combinado["TeamDivision"] = pd.NA
        combinado["TeamAbbr"] = pd.NA
        return combinado

    combinado = stats_df.copy()
    combinado = combinado.merge(equipos_df, on="Team", how="left")
    if "League" in combinado.columns:
        combinado["TeamLeague"] = combinado["TeamLeague"].fillna(combinado["League"])
    combinado["TeamDivision"] = combinado["TeamDivision"].fillna("Sin division")
    return combinado


def cargar_stats_2025():
    cache = _cargar_cache_df("stats_2025.csv")
    if cache is not None and not cache.empty:
        return cache, "Cache local (MLB/FanGraphs)"

    try:
        stats = cargar_stats(2025)
        if not stats.empty:
            _guardar_cache_df(stats, "stats_2025.csv")
            return stats, "MLB Stats API"
    except Exception as error_mlb_api:
        try:
            stats_fg = batting_stats(2025, qual=10)
            _guardar_cache_df(stats_fg, "stats_2025.csv")
            return stats_fg, "FanGraphs (pybaseball fallback)"
        except Exception as error_fangraphs:
            raise RuntimeError(
                "No se pudieron cargar las stats MLB 2025 desde ninguna fuente. "
                f"MLB API: {error_mlb_api} | "
                f"FanGraphs (pybaseball): {error_fangraphs}"
            ) from error_fangraphs


def cargar_precios_spotrac(season=2025):
    cache = _cargar_cache_df(f"precios_spotrac_{season}.csv")
    if cache is not None and not cache.empty:
        return cache

    url = f"https://www.spotrac.com/mlb/rankings/player/_/year/{season}/sort/cap_total"
    respuesta = _http_get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
    respuesta.raise_for_status()

    texto = BeautifulSoup(respuesta.text, "html.parser").get_text("\n")
    patron = re.compile(
        r"(?P<name>[A-ZÀ-ÿ][A-Za-zÀ-ÿ\'.\- ]+?)\s*\n+\s*(?P<team>[A-Z]{2,3}),\s*(?P<pos>[A-Z0-9/\- ]+)\s*\n+.*?\$\s*(?P<salary>[\d,]+)",
        re.S,
    )

    filas = []
    for match in patron.finditer(texto):
        filas.append({
            "Name": match.group("name").strip(),
            "SpotracTeam": match.group("team").strip(),
            "SpotracPos": match.group("pos").strip(),
            "Salary": int(match.group("salary").replace(",", "")),
        })

    precios = pd.DataFrame(filas)
    if precios.empty:
        return precios

    precios["name_key"] = precios["Name"].map(normalizar_nombre)
    _guardar_cache_df(precios, f"precios_spotrac_{season}.csv")
    return precios


def enriquecer_con_precios(stats_df, precios_df):
    if precios_df.empty:
        stats_df["Salary"] = pd.NA
        return stats_df

    combinado = stats_df.copy()
    combinado["name_key"] = combinado["Name"].map(normalizar_nombre)
    combinado = combinado.merge(
        precios_df[["name_key", "Salary", "SpotracTeam", "SpotracPos"]],
        on="name_key",
        how="left",
    )
    return combinado


def cargar_pitching_mlb(season=2025):
    cache = _cargar_cache_df(f"pitching_{season}.csv")
    if cache is not None and not cache.empty:
        return cache

    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1&limit=10000"
    )
    respuesta = _http_get(url, timeout=12)
    respuesta.raise_for_status()

    data = respuesta.json()
    stats = data.get("stats", [])
    if not stats:
        raise RuntimeError("La respuesta de MLB API no contiene bloque de pitchers.")

    filas = []
    for split in stats[0].get("splits", []):
        fila = {
            "Name": split.get("player", {}).get("fullName", ""),
            "Team": split.get("team", {}).get("name", ""),
            "League": split.get("league", {}).get("name", ""),
            "Pos": split.get("position", {}).get("abbreviation", "P"),
        }
        stat = split.get("stat", {})
        fila.update({
            "ERA": stat.get("era"),
            "IP": stat.get("inningsPitched"),
            "G": stat.get("gamesPitched"),
            "GS": stat.get("gamesStarted"),
            "SV": stat.get("saves"),
            "HLD": stat.get("holds"),
            "WHIP": stat.get("whip"),
            "K9": stat.get("strikeoutsPer9Inn"),
            "BB9": stat.get("walksPer9Inn"),
        })
        filas.append(fila)

    pitchers = pd.DataFrame(filas)
    if pitchers.empty:
        return pitchers

    pitchers["IP"] = pd.to_numeric(pitchers["IP"], errors="coerce")
    pitchers["ERA"] = pd.to_numeric(pitchers["ERA"], errors="coerce")
    pitchers["G"] = pd.to_numeric(pitchers["G"], errors="coerce")
    pitchers["GS"] = pd.to_numeric(pitchers["GS"], errors="coerce")
    pitchers["WHIP"] = pd.to_numeric(pitchers["WHIP"], errors="coerce")
    pitchers["name_key"] = pitchers["Name"].map(normalizar_nombre)
    _guardar_cache_df(pitchers, f"pitching_{season}.csv")
    return pitchers


def enriquecer_pitchers_con_precios(pitchers_df, precios_df):
    if pitchers_df.empty:
        return pitchers_df
    combinado = pitchers_df.copy()
    combinado = combinado.merge(
        precios_df[["name_key", "Salary", "SpotracTeam", "SpotracPos"]],
        on="name_key",
        how="left",
    )
    return combinado


def crear_roster():
    return {pos: None for pos in ROSTER_SLOT_ORDEN}


@lru_cache(maxsize=512)
def _slots_para_posicion(pos_texto):
    texto = str(pos_texto).upper()
    tokens = [token.strip() for token in re.split(r"[\s/,]+", texto) if token.strip()]
    if not tokens:
        return []

    if "P" in tokens or "SP" in tokens or "RP" in tokens:
        return ["P"]

    candidatos = []
    if "C" in tokens:
        candidatos.append("C")
    if "1B" in tokens:
        candidatos.append("1B")
    if "2B" in tokens:
        candidatos.append("2B")
    if "3B" in tokens:
        candidatos.append("3B")
    if "SS" in tokens:
        candidatos.append("SS")
    if "LF" in tokens:
        candidatos.append("LF")
    if "CF" in tokens:
        candidatos.append("CF")
    if "RF" in tokens:
        candidatos.append("RF")
    if "OF" in tokens:
        candidatos.extend(["LF", "CF", "RF"])
    if "DH" in tokens:
        candidatos.append("DH")
    return candidatos


def _slot_disponible_para_jugador(roster, jugador, es_pitcher=False):
    if es_pitcher:
        return None

    pos = jugador.get("Pos") or jugador.get("SpotracPos") or ""
    candidatos = _slots_para_posicion(pos)
    for slot in candidatos:
        if slot in roster and roster[slot] is None:
            return slot
    return None


def draftear_desde_resultado(resultado, roster, es_pitcher=False):
    if resultado.empty:
        print("\nNo hay jugadores para draftear.")
        return

    seleccion = input("\nEscribe el numero del jugador para draftearlo o Enter para volver: ").strip()
    if not seleccion:
        return
    if not seleccion.isdigit():
        print("\nSeleccion invalida.")
        return

    indice = int(seleccion) - 1
    if indice < 0 or indice >= len(resultado):
        print("\nEl numero esta fuera de rango.")
        return

    jugador = resultado.iloc[indice].to_dict()
    slot = _slot_disponible_para_jugador(roster, jugador, es_pitcher=es_pitcher)
    if slot is None:
        print("\nNo hay posicion disponible para ese jugador o esa posicion ya esta ocupada.")
        return

    roster[slot] = jugador
    print(f"\nDrafteado: {jugador.get('Name')} en la posicion {slot}.")


def _formatear_salario(valor):
    salario = _to_number(valor)
    if pd.isna(salario):
        return "N/A"
    return f"${int(salario):,}"


def _nombre_jugador(jugador):
    return str(jugador.get("Name", "")).strip().lower()


def _es_jugador_ocupado(roster, jugador):
    nombre = _nombre_jugador(jugador)
    for asignado in roster.values():
        if asignado is not None and _nombre_jugador(asignado) == nombre:
            return True
    return False


def _filtrar_hitter_disponible(stats_df, roster):
    candidatos = stats_df.copy()
    if "Name" in candidatos.columns:
        ocupados = {_nombre_jugador(j) for j in roster.values() if j is not None}
        candidatos = candidatos[~_normaliza_texto(candidatos["Name"]).isin(ocupados)].copy()
    if "WAR" in candidatos.columns:
        candidatos["WAR"] = pd.to_numeric(candidatos["WAR"], errors="coerce")
    if "Salary" in candidatos.columns:
        candidatos["Salary"] = pd.to_numeric(candidatos["Salary"], errors="coerce")
    return candidatos


def _sugerir_por_slot(stats_df, roster, slot, presupuesto_restante=None):
    candidatos = _filtrar_hitter_disponible(stats_df, roster)
    if candidatos.empty:
        return None

    columna_pos = None
    for nombre_columna in ("Pos", "Position", "POS", "Pos Summary", "pos"):
        if nombre_columna in candidatos.columns:
            columna_pos = nombre_columna
            break

    if columna_pos is None:
        filtro = candidatos
    else:
        posiciones = candidatos[columna_pos].astype(str)

        if slot == "DH":
            filtro = candidatos
        elif slot == "LF":
            filtro = candidatos[posiciones.str.contains(r"LF|OF", case=False, na=False)]
        elif slot == "CF":
            filtro = candidatos[posiciones.str.contains(r"CF|OF", case=False, na=False)]
        elif slot == "RF":
            filtro = candidatos[posiciones.str.contains(r"RF|OF", case=False, na=False)]
        else:
            filtro = candidatos[posiciones.str.contains(slot, case=False, na=False)]

    if filtro.empty:
        filtro = candidatos

    if presupuesto_restante is not None and "Salary" in filtro.columns:
        salarios = pd.to_numeric(filtro["Salary"], errors="coerce")
        filtro = filtro[salarios <= presupuesto_restante]

    if filtro.empty:
        filtro = candidatos
        if presupuesto_restante is not None and "Salary" in filtro.columns:
            salarios = pd.to_numeric(filtro["Salary"], errors="coerce")
            filtro = filtro[salarios <= presupuesto_restante]

    if "WAR" in filtro.columns:
        filtro = filtro.sort_values(by=["WAR", "Salary"], ascending=[False, True])
    else:
        filtro = filtro.head(1)

    return filtro.head(1)


def mostrar_sugerencias_posiciones(stats_df, roster, presupuesto_restante=None):
    faltantes = [slot for slot in ROSTER_SLOT_ORDEN if roster.get(slot) is None]
    if not faltantes:
        print("\nTu roster ya esta completo.")
        return

    print("\n=== SUGERENCIAS PARA POSICIONES FALTANTES ===")
    for slot in faltantes:
        sugerido = _sugerir_por_slot(stats_df, roster, slot, presupuesto_restante)
        if sugerido is None or sugerido.empty:
            print(f"{slot}: sin sugerencia disponible")
            continue
        jugador = sugerido.iloc[0]
        war = jugador.get("WAR", "N/A")
        salary = _formatear_salario(jugador.get("Salary", "N/A"))
        print(f"{slot}: {jugador.get('Name')} ({jugador.get('Team', '-')}) - WAR {war} - {salary}")


def mostrar_roster(roster, stats_df=None, presupuesto_restante=None, limpiar=True):
    if limpiar:
        limpiar_pantalla()
    filas = []
    for slot in ROSTER_SLOT_ORDEN:
        jugador = roster.get(slot)
        if jugador is None:
            filas.append({"Slot": slot, "Name": "VACIO", "Team": "-", "Salary": "-", "WAR": "-"})
            continue
        fila = {
            "Slot": slot,
            "Name": jugador.get("Name", ""),
            "Team": jugador.get("Team", ""),
            "Salary": _formatear_salario(jugador.get("Salary", "N/A")),
            "WAR": jugador.get("WAR", jugador.get("PitchProxyWAR", "N/A")),
        }
        filas.append(fila)

    df = pd.DataFrame(filas)
    if df.empty:
        print("\nNo hay roster para mostrar.")
        return

    print("\n=== TU ROSTER ===")
    if presupuesto_restante is not None:
        print(f"Presupuesto restante: ${presupuesto_restante:,.0f}")
    print(df.to_string(index=False))

    if stats_df is not None:
        mostrar_sugerencias_posiciones(stats_df, roster, presupuesto_restante)


def _valor_hitter(jugador):
    war = _to_number(jugador.get("WAR"))
    salary = _to_number(jugador.get("Salary"))
    if pd.isna(war) or pd.isna(salary) or salary <= 0:
        return 0
    return float(war) / (salary / 1_000_000)


def _valor_pitcher(jugador):
    era = _to_number(jugador.get("ERA"))
    ip = _to_number(jugador.get("IP"))
    salary = _to_number(jugador.get("Salary"))
    if pd.isna(era) or pd.isna(ip) or pd.isna(salary) or salary <= 0:
        return 0
    proxy = max(0, (4.20 - era)) * (ip / 45.0)
    return proxy / (salary / 1_000_000)


def mostrar_pitchers_valiosos(pitchers_df):
    if pitchers_df.empty:
        print("\nNo hay datos de pitchers disponibles.")
        return pitchers_df

    vista = pitchers_df.copy()
    vista = vista[pd.notna(vista["ERA"]) & pd.notna(vista["IP"])].copy()
    vista = vista[vista["IP"] >= 20].copy()
    vista["Value"] = vista.apply(_valor_pitcher, axis=1)
    vista = vista.sort_values(by=["Value", "ERA"], ascending=[False, True]).head(25)

    columnas = [c for c in ["Name", "Team", "Pos", "ERA", "IP", "Salary", "WHIP", "Value"] if c in vista.columns]
    tabla = vista[columnas].copy()
    if "Salary" in tabla.columns:
        tabla["Salary"] = tabla["Salary"].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "N/A")
    if "Value" in tabla.columns:
        tabla["Value"] = tabla["Value"].round(2)

    tabla = tabla.reset_index(drop=True)
    tabla.insert(0, "#", tabla.index + 1)

    print("\n=== MEJORES PITCHERS POR ERA Y PRECIO ===")
    print(tabla.to_string(index=False))
    return vista.reset_index(drop=True)


def construir_bullpen_por_equipo(pitchers_df):
    if pitchers_df.empty:
        return pd.DataFrame()

    bullpen = pitchers_df.copy()
    bullpen = bullpen[pd.to_numeric(bullpen["GS"], errors="coerce").fillna(0) == 0].copy()
    bullpen["IP"] = pd.to_numeric(bullpen["IP"], errors="coerce")
    bullpen["ERA"] = pd.to_numeric(bullpen["ERA"], errors="coerce")
    bullpen = bullpen[pd.notna(bullpen["IP"]) & pd.notna(bullpen["ERA"])].copy()

    if bullpen.empty:
        return bullpen

    liga_era = bullpen["ERA"].mean()
    grupos = []
    for team, grupo in bullpen.groupby("Team"):
        ip_total = grupo["IP"].sum()
        if ip_total <= 0:
            continue
        era_equipo = (grupo["ERA"] * grupo["IP"]).sum() / ip_total
        grupos.append({
            "Team": team,
            "BullpenERA": round(era_equipo, 2),
            "BullpenIP": round(ip_total, 1),
            "BullpenScore": round(liga_era - era_equipo, 2),
        })

    return pd.DataFrame(grupos).sort_values(by="BullpenScore", ascending=False).reset_index(drop=True)


def _fuerza_equipo_draft(roster):
    total_war = 0.0
    pitcher_proxy = 0.0

    for slot, jugador in roster.items():
        if jugador is None:
            continue
        if slot == "P":
            pitcher_proxy += float(jugador.get("PitchProxyWAR", 0) or 0)
        else:
            war = _to_number(jugador.get("WAR"))
            if not pd.isna(war):
                total_war += float(war)

    return total_war + pitcher_proxy


def _simular_series(strength_a, strength_b, mejor_de=5):
    victorias_necesarias = mejor_de // 2 + 1
    wins_a = 0
    wins_b = 0

    while wins_a < victorias_necesarias and wins_b < victorias_necesarias:
        prob_a = 1 / (1 + math.exp(-(strength_a - strength_b) / 5.0))
        if random.random() < prob_a:
            wins_a += 1
        else:
            wins_b += 1

    return "A" if wins_a > wins_b else "B"


def construir_roster_optimo_con_presupuesto(stats_df, presupuesto_total, equipo_preferido=None):
    candidatos = stats_df.copy()
    if "Salary" not in candidatos.columns or "WAR" not in candidatos.columns:
        return None, 0

    candidatos["Salary"] = pd.to_numeric(candidatos["Salary"], errors="coerce")
    candidatos["WAR"] = pd.to_numeric(candidatos["WAR"], errors="coerce")
    candidatos = candidatos[pd.notna(candidatos["Salary"]) & pd.notna(candidatos["WAR"])].copy()
    candidatos = candidatos[candidatos["Salary"] > 0].copy()
    if candidatos.empty:
        return None, 0

    candidatos["base_pos"] = candidatos["Pos"].astype(str).str.split("/").str[0].str.upper()
    if equipo_preferido:
        objetivo = _normalizar_equipo_texto(equipo_preferido)
        team_norm = candidatos.get("Team", pd.Series("", index=candidatos.index)).astype(str).map(_normalizar_equipo_texto)
        coincide = (team_norm.str.contains(re.escape(objetivo), na=False)) | (team_norm.map(lambda x: bool(x) and x in objetivo))
        candidatos["TeamPriority"] = (~coincide).astype(int)
    else:
        candidatos["TeamPriority"] = 0
    slot_candidatos = {}

    def _ranking_candidatos(subset):
        if subset.empty:
            return subset

        vista = subset.copy()
        vista["ValueScore"] = (vista["WAR"] / (vista["Salary"] / 1_000_000)).replace([float("inf"), -float("inf")], 0).fillna(0)

        top_war = vista.sort_values(
            by=["TeamPriority", "WAR", "Salary"],
            ascending=[True, False, True],
        ).head(60)
        top_baratos = vista.sort_values(
            by=["TeamPriority", "Salary", "WAR"],
            ascending=[True, True, False],
        ).head(60)
        top_valor = vista.sort_values(
            by=["TeamPriority", "ValueScore", "WAR", "Salary"],
            ascending=[True, False, False, True],
        ).head(60)

        combinados = pd.concat([top_war, top_baratos, top_valor], ignore_index=True)
        combinados = combinados.drop_duplicates(subset=["Name"]).reset_index(drop=True)
        return combinados.sort_values(
            by=["TeamPriority", "ValueScore", "WAR", "Salary"],
            ascending=[True, False, False, True],
        ).head(120).reset_index(drop=True)

    for slot in ROSTER_SLOT_ORDEN:
        if slot == "DH":
            subset = candidatos.copy()
        elif slot in ("LF", "CF", "RF"):
            subset = candidatos[candidatos["Pos"].astype(str).str.contains(rf"{slot}|OF", case=False, na=False, regex=True)].copy()
        else:
            subset = candidatos[(candidatos["Pos"].astype(str).str.contains(rf"\b{slot}\b", case=False, na=False, regex=True)) | (candidatos["base_pos"] == slot)].copy()

        slot_candidatos[slot] = _ranking_candidatos(subset)

    orden = sorted(ROSTER_SLOT_ORDEN, key=lambda slot: len(slot_candidatos[slot]) if len(slot_candidatos[slot]) else 999)
    max_war_slot = {
        slot: (slot_candidatos[slot]["WAR"].max() if not slot_candidatos[slot].empty else 0)
        for slot in ROSTER_SLOT_ORDEN
    }

    mejor_roster = None
    mejor_war = -1.0

    def es_compatible(jugador, slot):
        posicion = str(jugador.get("Pos", "")).upper()
        base_pos = posicion.split("/")[0]
        if slot == "DH":
            return True
        if slot in ("LF", "CF", "RF"):
            return slot in posicion or "OF" in posicion or base_pos == slot
        return slot in posicion or base_pos == slot

    def backtrack(indice, roster_actual, usados, gasto_actual, war_actual):
        nonlocal mejor_roster, mejor_war

        if gasto_actual > presupuesto_total:
            return

        if indice == len(orden):
            if war_actual > mejor_war:
                mejor_war = war_actual
                mejor_roster = roster_actual.copy()
            return

        techo_restante = sum(max_war_slot[slot] for slot in orden[indice:])
        if war_actual + techo_restante < mejor_war:
            return

        slot = orden[indice]
        candidatos_slot = slot_candidatos[slot]
        if candidatos_slot.empty:
            return

        for _, jugador in candidatos_slot.iterrows():
            nombre = jugador["Name"]
            salario = float(jugador["Salary"])
            war = float(jugador["WAR"])
            if nombre in usados:
                continue
            if gasto_actual + salario > presupuesto_total:
                continue
            if not es_compatible(jugador, slot):
                continue

            roster_actual[slot] = jugador.to_dict()
            usados.add(nombre)
            backtrack(indice + 1, roster_actual, usados, gasto_actual + salario, war_actual + war)
            usados.remove(nombre)
            roster_actual.pop(slot, None)

    backtrack(0, {}, set(), 0.0, 0.0)

    if not mejor_roster:
        # Fallback: intenta completar usando los mas baratos por slot, aunque no sea WAR-optimo.
        roster_fallback = {}
        usados_fallback = set()
        gasto_fallback = 0.0

        for slot in orden:
            candidatos_slot = slot_candidatos[slot]
            if candidatos_slot.empty:
                return None, 0

            elegible = candidatos_slot[~candidatos_slot["Name"].isin(usados_fallback)].copy()
            elegible = elegible[elegible.apply(lambda row: es_compatible(row, slot), axis=1)]
            if elegible.empty:
                return None, 0

            dentro_presupuesto = elegible[elegible["Salary"] <= (presupuesto_total - gasto_fallback)]
            if not dentro_presupuesto.empty:
                pick = dentro_presupuesto.sort_values(by=["ValueScore", "WAR", "Salary"], ascending=[False, False, True]).iloc[0]
            else:
                pick = elegible.sort_values(by=["Salary", "WAR"], ascending=[True, False]).iloc[0]

            roster_fallback[slot] = pick.to_dict()
            usados_fallback.add(pick["Name"])
            gasto_fallback += float(pick["Salary"])

        mejor_roster = roster_fallback

    roster = crear_roster()
    gasto_total = 0
    for slot in ROSTER_SLOT_ORDEN:
        jugador = mejor_roster.get(slot)
        if jugador is None:
            continue
        roster[slot] = jugador
        salario = _to_number(jugador.get("Salary"))
        if pd.notna(salario):
            gasto_total += int(salario)

    return roster, gasto_total


def simular_temporada_y_playoffs(roster, pitchers_df, bullpen_df):
    if any(slot is None for slot in roster.values()):
        print("\nTu equipo no esta completo. Completa las 9 posiciones antes de simular.")
        return

    fuerza_base = _fuerza_equipo_draft(roster)
    bullpen_df = bullpen_df.copy()
    if bullpen_df.empty:
        print("\nNo se pudo construir el bullpen de la liga para la simulacion.")
        return

    bullpen_df["BullpenScore"] = pd.to_numeric(bullpen_df["BullpenScore"], errors="coerce").fillna(0)
    bullpen_df = bullpen_df.sort_values(by="BullpenScore", ascending=False).reset_index(drop=True)

    wins_estimados = []
    for _, fila in bullpen_df.iterrows():
        score = 81 + (fila["BullpenScore"] * 3.5) + random.gauss(0, 4)
        wins_estimados.append({
            "Team": fila["Team"],
            "Wins": int(round(score)),
            "BullpenScore": float(fila["BullpenScore"]),
        })

    nuestro_equipo = {
        "Team": "TU EQUIPO",
        "Wins": int(round(81 + (fuerza_base - 30) + random.gauss(0, 3))),
        "BullpenScore": 0.0,
        "Strength": fuerza_base,
    }

    liga = pd.DataFrame(wins_estimados)
    liga = pd.concat([liga, pd.DataFrame([{"Team": nuestro_equipo["Team"], "Wins": nuestro_equipo["Wins"], "BullpenScore": 0.0}])], ignore_index=True)
    liga = liga.sort_values(by="Wins", ascending=False).reset_index(drop=True)

    print("\n=== SIMULACION DE TEMPORADA MLB ===")
    print(liga.head(15).to_string(index=False))

    posicion = liga.index[liga["Team"] == "TU EQUIPO"].tolist()[0] + 1
    print(f"\nTu equipo termino en la posicion {posicion} con {nuestro_equipo['Wins']} victorias.")

    top_12 = liga.head(12)["Team"].tolist()
    if "TU EQUIPO" not in top_12:
        print("No alcanzo playoffs.")
        return

    print("Entraste a playoffs.")
    strength_map = {fila["Team"]: 30 + (fila["BullpenScore"] * 3.5) for _, fila in bullpen_df.iterrows()}
    strength_map["TU EQUIPO"] = fuerza_base

    seed_order = liga["Team"].head(12).tolist()
    if "TU EQUIPO" in seed_order:
        print(f"Tu semilla de playoffs es #{seed_order.index('TU EQUIPO') + 1}.")

    actual = "TU EQUIPO"
    ronda = "Wild Card"
    for fase, mejor_de in [("Wild Card", 3), ("Division Series", 5), ("League Championship Series", 7), ("World Series", 7)]:
        rivales = [team for team in seed_order if team != actual]
        if not rivales:
            break
        rival = rivales[0]
        ganador = _simular_series(strength_map.get(actual, 30), strength_map.get(rival, 30), mejor_de=mejor_de)
        if ganador != "A":
            print(f"Eliminado en {fase} contra {rival}.")
            return
        print(f"Ganaste {fase} contra {rival}.")

    print("Campeon de la Serie Mundial.")


def completar_stats_avanzadas_aprox(df, fuente):
    vista = df.copy()
    requeridas = {
        "atBats", "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies",
        "hits", "doubles", "triples", "homeRuns", "plateAppearances"
    }
    if not requeridas.issubset(set(vista.columns)):
        # Si no hay columnas base para aproximar, conserva dataset tal cual.
        if "WAR" in vista.columns:
            vista["WAR"] = pd.to_numeric(vista["WAR"], errors="coerce")
        return vista, False

    cols_num = list(requeridas) + ["stolenBases", "caughtStealing"]
    for col in cols_num:
        if col in vista.columns:
            vista[col] = pd.to_numeric(vista[col], errors="coerce")

    ab = vista["atBats"]
    bb = vista["baseOnBalls"]
    ibb = vista["intentionalWalks"].fillna(0)
    hbp = vista["hitByPitch"]
    sf = vista["sacFlies"].fillna(0)
    hits = vista["hits"]
    doubles = vista["doubles"]
    triples = vista["triples"]
    hr = vista["homeRuns"]
    pa = vista["plateAppearances"]

    singles = hits - doubles - triples - hr
    ubb = bb - ibb
    denom = ab + bb - ibb + sf + hbp

    # Coeficientes aproximados de wOBA para mantener continuidad cuando FanGraphs no responde.
    woba_num = (0.69 * ubb) + (0.72 * hbp) + (0.88 * singles) + (1.247 * doubles) + (1.578 * triples) + (2.031 * hr)
    woba = woba_num / denom.replace(0, pd.NA)

    total_pa = pa.sum(skipna=True)
    if total_pa and total_pa > 0:
        lg_woba = (woba.fillna(0) * pa.fillna(0)).sum() / total_pa
    else:
        lg_woba = 0.315

    woba_scale = 1.25
    wraa = ((woba - lg_woba) / woba_scale) * pa

    if "wOBA" not in vista.columns:
        vista["wOBA"] = woba.round(3)
    else:
        vista["wOBA"] = pd.to_numeric(vista["wOBA"], errors="coerce").fillna(woba.round(3))

    if "wRAA" not in vista.columns:
        vista["wRAA"] = wraa.round(1)
    else:
        vista["wRAA"] = pd.to_numeric(vista["wRAA"], errors="coerce").fillna(wraa.round(1))

    if "Batting" not in vista.columns:
        vista["Batting"] = vista["wRAA"]
    else:
        vista["Batting"] = pd.to_numeric(vista["Batting"], errors="coerce").fillna(vista["wRAA"])

    # Ajuste posicional aproximado por temporada completa (162 juegos), prorateado.
    ajustes_posicionales = {
        "C": 12.5,
        "1B": -12.5,
        "2B": 2.5,
        "3B": 2.5,
        "SS": 7.5,
        "LF": -7.5,
        "CF": 2.5,
        "RF": -7.5,
        "DH": -17.5,
        "OF": -2.5,
        "UTIL": -17.5,
    }

    if "Pos" in vista.columns:
        pos_abbr = vista["Pos"].astype(str).str.split("/").str[0].str.upper()
        ajuste_por_jugador = pos_abbr.map(ajustes_posicionales).fillna(0)
    else:
        ajuste_por_jugador = 0

    if "gamesPlayed" in vista.columns:
        juegos = pd.to_numeric(vista["gamesPlayed"], errors="coerce").fillna(0)
    elif "G" in vista.columns:
        juegos = pd.to_numeric(vista["G"], errors="coerce").fillna(0)
    else:
        juegos = 0
    vista["Positional"] = (ajuste_por_jugador * (juegos / 162.0)).round(1)

    # BaseRunning y Fielding no vienen en MLB API de esta ruta; se inicializan en 0 como aproximacion.
    if "BaseRunning" not in vista.columns:
        vista["BaseRunning"] = 0.0
    else:
        vista["BaseRunning"] = pd.to_numeric(vista["BaseRunning"], errors="coerce").fillna(0.0)

    if "Fielding" not in vista.columns:
        vista["Fielding"] = 0.0
    else:
        vista["Fielding"] = pd.to_numeric(vista["Fielding"], errors="coerce").fillna(0.0)

    # Estimacion simplificada de WAR en fallback.
    replacement_runs = 20.0 * (pa / 600.0)
    runs_per_win = 10.0
    off_total = vista["Batting"] + vista["BaseRunning"] + vista["Positional"]
    def_total = vista["Fielding"]
    war_aprox = ((off_total + def_total + replacement_runs) / runs_per_win).round(1)

    if "Off" not in vista.columns:
        vista["Off"] = off_total.round(1)
    else:
        vista["Off"] = pd.to_numeric(vista["Off"], errors="coerce").fillna(off_total.round(1))

    if "Def" not in vista.columns:
        vista["Def"] = def_total.round(1)
    else:
        vista["Def"] = pd.to_numeric(vista["Def"], errors="coerce").fillna(def_total.round(1))

    if "WAR" not in vista.columns:
        vista["WAR"] = war_aprox
    else:
        vista["WAR"] = pd.to_numeric(vista["WAR"], errors="coerce").fillna(war_aprox)

    # Evita NaN residuales en salida final para menus/roster.
    vista["WAR"] = pd.to_numeric(vista["WAR"], errors="coerce").fillna(0.0).round(1)

    return vista, True

def _buscar_columna(df, candidatas):
    for columna in candidatas:
        if columna in df.columns:
            return columna
    return None


def _normaliza_texto(serie):
    return serie.astype(str).str.strip().str.lower()


def buscar_stats(df, nombre="", equipo="", liga="", posicion="", pais=""):
    resultado = df.copy()

    columna_nombre = _buscar_columna(resultado, ('Name', 'player', 'Player'))
    columna_equipo = _buscar_columna(resultado, ('Team', 'Tm', 'team'))
    columna_liga = _buscar_columna(resultado, ('League', 'Lg', 'league'))

    if nombre and columna_nombre:
        resultado = resultado[_normaliza_texto(resultado[columna_nombre]).str.contains(nombre.strip().lower(), na=False)]
    elif nombre and not columna_nombre:
        print("No existe una columna de nombre en los datos cargados.")

    if equipo and columna_equipo:
        resultado = resultado[_normaliza_texto(resultado[columna_equipo]).str.contains(equipo.strip().lower(), na=False)]
    elif equipo and not columna_equipo:
        print("No existe una columna de equipo en los datos cargados.")

    if liga and columna_liga:
        resultado = resultado[_normaliza_texto(resultado[columna_liga]) == liga.strip().lower()]
    elif liga and not columna_liga:
        print("No existe una columna de liga en los datos cargados.")

    # En pybaseball, la posicion puede venir como 'Pos' dependiendo de la tabla.
    columna_posicion = None
    for c in ('Pos', 'Position', 'POS', 'Pos Summary', 'pos'):
        if c in resultado.columns:
            columna_posicion = c
            break

    if posicion and columna_posicion:
        resultado = resultado[_normaliza_texto(resultado[columna_posicion]).str.contains(posicion.strip().lower(), na=False)]
    elif posicion and not columna_posicion:
        print("No existe una columna de posicion en los datos cargados.")

    columna_pais = _buscar_columna(resultado, ("Country", "BirthCountry", "birthCountry", "Pais", "CountryOfBirth"))
    if pais and columna_pais:
        resultado = resultado[_normaliza_texto(resultado[columna_pais]).str.contains(pais.strip().lower(), na=False)]
    elif pais and not columna_pais:
        print("No existe una columna de pais en los datos cargados.")

    return resultado


def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def obtener_columnas(df):
    equivalencias = {
        "PA": "plateAppearances",
        "G": "gamesPlayed",
        "AB": "atBats",
        "H": "hits",
        "HR": "homeRuns",
        "R": "runs",
        "RBI": "rbi",
        "AVG": "avg",
        "OBP": "obp",
        "SLG": "slg",
        "OPS": "ops",
    }
    columnas_jugador = ["Name", "Team", "TeamLeague", "TeamDivision", "Country", "Salary", "League", "Pos"]
    stats_basicos = ["G", "PA", "AB", "H", "HR", "R", "RBI", "AVG", "OBP", "SLG", "OPS"]

    disponibles = []

    for col in columnas_jugador + stats_basicos + columnas_interes:
        if col in df.columns:
            disponibles.append(col)
        elif col in equivalencias and equivalencias[col] in df.columns:
            disponibles.append(equivalencias[col])

    # Elimina duplicados preservando el orden.
    disponibles = list(dict.fromkeys(disponibles))

    if disponibles:
        return disponibles
    return df.columns.tolist()


def mostrar_resultado_limpio(resultado, con_rango=False, limite=30):
    limpiar_pantalla()
    if resultado.empty:
        print("\nNo se encontraron jugadores con ese filtro.")
        return

    vista = resultado.copy()
    columnas_numericas = vista.select_dtypes(include="number").columns
    for col in columnas_numericas:
        vista[col] = vista[col].round(3)

    if "WAR" in vista.columns:
        vista = vista.sort_values(by="WAR", ascending=False)

    columnas = obtener_columnas(vista)
    stats_avanzados = ["PA", "wOBA", "wRAA", "Batting", "BaseRunning", "Fielding", "Positional", "Off", "Def", "WAR"]

    equivalencias = {
        "PA": "plateAppearances",
    }

    # Agrega forzosamente columnas avanzadas (o su equivalente) para que siempre salgan en la tabla.
    for stat in stats_avanzados:
        if stat in vista.columns and stat not in columnas:
            columnas.append(stat)
        elif stat in equivalencias and equivalencias[stat] in vista.columns:
            if equivalencias[stat] not in columnas:
                columnas.append(equivalencias[stat])
        elif stat not in vista.columns:
            vista[stat] = "N/A"
            if stat not in columnas:
                columnas.append(stat)

    columnas = list(dict.fromkeys(columnas))
    encabezados = {
        "Name": "NAME",
        "Team": "TEAM",
        "TeamLeague": "TEAM LEAGUE",
        "TeamDivision": "TEAM DIVISION",
        "Country": "COUNTRY",
        "Salary": "SALARY",
        "SpotracTeam": "SPOTRAC TEAM",
        "SpotracPos": "SPOTRAC POS",
        "League": "LEAGUE",
        "Pos": "POS",
        "G": "G",
        "gamesPlayed": "GAMES PLAYED",
        "PA": "PA",
        "plateAppearances": "PA",
        "AB": "AB",
        "atBats": "AB",
        "H": "H",
        "hits": "H",
        "HR": "HR",
        "homeRuns": "HR",
        "R": "R",
        "runs": "R",
        "RBI": "RBI",
        "rbi": "RBI",
        "AVG": "AVG",
        "avg": "AVG",
        "OBP": "OBP",
        "obp": "OBP",
        "SLG": "SLG",
        "slg": "SLG",
        "OPS": "OPS",
        "ops": "OPS",
        "wOBA": "WOBA",
        "wRAA": "WRAA",
        "Batting": "BATTING",
        "BaseRunning": "BASERUNNING",
        "Fielding": "FIELDING",
        "Positional": "POSITIONAL",
        "Off": "OFF",
        "Def": "DEF",
        "WAR": "WAR",
    }

    tabla_mostrar = vista[columnas].head(limite).copy()
    tabla_mostrar = tabla_mostrar.rename(columns=lambda c: encabezados.get(c, str(c).upper()))

    if con_rango:
        tabla_mostrar.insert(0, "#", range(1, len(tabla_mostrar) + 1))

    print(f"\nResultados encontrados: {len(vista)}")
    print(tabla_mostrar.to_string(index=False))
    if len(vista) > limite:
        print(f"\nMostrando los primeros {limite} resultados.")

    return tabla_mostrar


def mostrar_equipos_disponibles(stats_df):
    limpiar_pantalla()
    columna_equipo = _buscar_columna(stats_df, ("Team", "Tm", "team"))
    if columna_equipo is None:
        print("\nNo hay una columna de equipos disponible.")
        return []

    columnas_meta = [c for c in ("TeamLeague", "TeamDivision") if c in stats_df.columns]
    if len(columnas_meta) < 2:
        print("\nNo hay metadatos suficientes para dividir equipos por liga y division.")
        equipos = sorted(
            {
                str(equipo).strip()
                for equipo in stats_df[columna_equipo].dropna().astype(str).tolist()
                if str(equipo).strip()
            }
        )
        for indice, equipo in enumerate(equipos, start=1):
            print(f"{indice:>2}. {equipo}")
        return equipos

    vista = stats_df[["Team", "TeamLeague", "TeamDivision"]].dropna(subset=["Team", "TeamLeague", "TeamDivision"]).drop_duplicates().copy()
    vista = vista.sort_values(by=["TeamLeague", "TeamDivision", "Team"])

    print("\n=== EQUIPOS DISPONIBLES POR LIGA Y DIVISION ===")
    equipos = []
    for league, grupos_liga in vista.groupby("TeamLeague", dropna=False):
        liga_label = str(league).strip() if pd.notna(league) and str(league).strip() else "Sin liga"
        print(f"\nLiga: {liga_label}")
        for division, grupos_division in grupos_liga.groupby("TeamDivision", dropna=False):
            division_label = str(division).strip() if pd.notna(division) and str(division).strip() else "Sin division"
            print(f"  Division: {division_label}")
            tabla = grupos_division[["Team"]].drop_duplicates().reset_index(drop=True)
            tabla.insert(0, "#", tabla.index + 1)
            print(tabla.to_string(index=False))
            equipos.extend(grupos_division["Team"].tolist())

    equipos = list(dict.fromkeys(equipos))
    return equipos


def mostrar_leaderboard_hitters(stats_df, roster, presupuesto_restante, limite=25):
    limpiar_pantalla()
    if stats_df.empty:
        print("\nNo hay datos de hitters disponibles.")
        return presupuesto_restante

    vista = _filtrar_hitter_disponible(stats_df, roster)
    if vista.empty:
        print("\nNo quedan hitters disponibles para mostrar.")
        return presupuesto_restante

    if "WAR" in vista.columns:
        vista = vista.sort_values(by=["WAR", "Salary"], ascending=[False, True])

    columnas = [c for c in ["Name", "Team", "Pos", "Salary", "WAR", "G", "PA", "AVG", "OBP", "SLG", "OPS"] if c in vista.columns]
    tabla = vista[columnas].head(limite).copy()
    if "Salary" in tabla.columns:
        tabla["Salary"] = tabla["Salary"].apply(_formatear_salario)
    if "WAR" in tabla.columns:
        tabla["WAR"] = pd.to_numeric(tabla["WAR"], errors="coerce").round(1)
    if "AVG" in tabla.columns:
        tabla["AVG"] = pd.to_numeric(tabla["AVG"], errors="coerce").round(3)
    if "OBP" in tabla.columns:
        tabla["OBP"] = pd.to_numeric(tabla["OBP"], errors="coerce").round(3)
    if "SLG" in tabla.columns:
        tabla["SLG"] = pd.to_numeric(tabla["SLG"], errors="coerce").round(3)
    if "OPS" in tabla.columns:
        tabla["OPS"] = pd.to_numeric(tabla["OPS"], errors="coerce").round(3)

    tabla = tabla.reset_index(drop=True)
    tabla.insert(0, "#", tabla.index + 1)

    print("\n=== LEADERBOARD DE HITTERS ===")
    print(f"Presupuesto restante: ${presupuesto_restante:,.0f}")
    print(tabla.to_string(index=False))
    mostrar_sugerencias_posiciones(stats_df, roster, presupuesto_restante)

    compra = input("\nNumero para draftear o Enter para volver: ").strip()
    if not compra:
        return presupuesto_restante
    if not compra.isdigit():
        print("\nSeleccion invalida.")
        return presupuesto_restante

    indice = int(compra) - 1
    if indice < 0 or indice >= len(tabla):
        print("\nEl numero esta fuera de rango.")
        return presupuesto_restante

    jugador = vista.head(limite).reset_index(drop=True).iloc[indice].to_dict()
    slot = _slot_disponible_para_jugador(roster, jugador, es_pitcher=False)
    if slot is None:
        print("\nNo hay posicion disponible para ese jugador o esa posicion ya esta ocupada.")
        return presupuesto_restante

    salario = _to_number(jugador.get("Salary"))
    if pd.isna(salario) or salario <= 0:
        print("\nEse jugador no tiene salario valido para el draft.")
        return presupuesto_restante
    if salario > presupuesto_restante:
        print(f"\nNo alcanza el presupuesto para {jugador.get('Name')}.")
        return presupuesto_restante

    roster[slot] = jugador
    presupuesto_restante -= int(salario)
    print(f"\nDrafteado: {jugador.get('Name')} en la posicion {slot}.")
    print(f"Presupuesto restante: ${presupuesto_restante:,.0f}")
    return presupuesto_restante


def ofrecer_draft_desde_resultado(resultado, roster, presupuesto_restante, es_pitcher=False):
    if resultado.empty:
        return presupuesto_restante

    eleccion = input("\nNumero para draftear o Enter para volver: ").strip()
    if not eleccion:
        return presupuesto_restante
    if not eleccion.isdigit():
        print("\nSeleccion invalida.")
        return presupuesto_restante

    indice = int(eleccion) - 1
    if indice < 0 or indice >= len(resultado):
        print("\nEl numero esta fuera de rango.")
        return presupuesto_restante

    jugador = resultado.iloc[indice].to_dict()
    slot = _slot_disponible_para_jugador(roster, jugador, es_pitcher=es_pitcher)
    if slot is None:
        print("\nNo hay posicion disponible para ese jugador o esa posicion ya esta ocupada.")
        return presupuesto_restante

    salario = _to_number(jugador.get("Salary"))
    if not es_pitcher and (pd.isna(salario) or salario <= 0):
        print("\nEse jugador no tiene salario valido para el draft.")
        return presupuesto_restante
    if not es_pitcher and salario > presupuesto_restante:
        print(f"\nNo alcanza el presupuesto para {jugador.get('Name')}.")
        return presupuesto_restante

    roster[slot] = jugador
    if not es_pitcher:
        presupuesto_restante -= int(salario)
        print(f"\nDrafteado: {jugador.get('Name')} en la posicion {slot}.")
        print(f"Presupuesto restante: ${presupuesto_restante:,.0f}")
    else:
        print(f"\nDrafteado: {jugador.get('Name')} en la posicion {slot}.")

    return presupuesto_restante


def simular_draft_9(stats_df):
    if "Salary" not in stats_df.columns:
        print("\nNo hay precios cargados para simular el draft.")
        return

    presupuesto_raw = input("Presupuesto total para 9 jugadores (ej. 50000000): ").strip()
    presupuesto = int(presupuesto_raw or "50000000")

    candidatos = stats_df.copy()
    candidatos = candidatos[pd.notna(candidatos["Salary"])].copy()
    if candidatos.empty:
        print("\nNo hay jugadores con salario disponible para armar el draft.")
        return

    if "WAR" not in candidatos.columns:
        candidatos["WAR"] = 0

    candidatos["Salary"] = pd.to_numeric(candidatos["Salary"], errors="coerce")
    candidatos = candidatos[pd.notna(candidatos["Salary"])].copy()
    candidatos["Value"] = candidatos["WAR"].fillna(0) / (candidatos["Salary"] / 1_000_000)
    candidatos = candidatos.sort_values(by=["Value", "WAR"], ascending=False)

    seleccionados = []
    gastado = 0
    for _, jugador in candidatos.iterrows():
        if len(seleccionados) >= 9:
            break
        salario = int(jugador["Salary"])
        if gastado + salario <= presupuesto:
            seleccionados.append(jugador)
            gastado += salario

    if len(seleccionados) < 9:
        print(f"\nNo fue posible completar 9 jugadores dentro del presupuesto de ${presupuesto:,.0f}.")
        print(f"Se seleccionaron {len(seleccionados)} jugadores.")
    else:
        print(f"\nDraft completado con 9 jugadores dentro del presupuesto de ${presupuesto:,.0f}.")

    draft_df = pd.DataFrame(seleccionados)
    if draft_df.empty:
        return

    columnas_draft = [c for c in ["Name", "Team", "Pos", "Salary", "WAR", "Value"] if c in draft_df.columns]
    draft_df = draft_df[columnas_draft].copy()
    draft_df["Salary"] = draft_df["Salary"].map(lambda x: f"${int(x):,}")
    if "Value" in draft_df.columns:
        draft_df["Value"] = draft_df["Value"].round(2)

    print(draft_df.to_string(index=False))
    print(f"\nTotal gastado: ${gastado:,.0f}")
    if "WAR" in draft_df.columns:
        total_war = pd.to_numeric(draft_df["WAR"], errors="coerce").fillna(0).sum()
        print(f"WAR proyectado: {total_war:.1f}")


def mostrar_inicio(roster, presupuesto_restante):
    limpiar_pantalla()
    print("=== SIMULADOR MLB 2025 ===")
    print("Arma un roster de hitters, controla el presupuesto y simula tu temporada.")
    print("Busca por nombre, equipo, liga, posicion o pais. En equipos veras la lista antes de elegir.")
    print(f"Presupuesto disponible: ${presupuesto_restante:,.0f}")
    faltantes = [slot for slot in ROSTER_SLOT_ORDEN if roster.get(slot) is None]
    print(f"Posiciones faltantes: {', '.join(faltantes) if faltantes else 'ninguna'}\n")


def menu_busqueda_bateadores(stats_2025, roster, fuente, presupuesto_restante):
    while True:
        limpiar_pantalla()
        print("=== BUSCADOR DE BATEADORES 2025 ===")
        print(f"Fuente de datos: {fuente}")
        print(f"Presupuesto restante: ${presupuesto_restante:,.0f}\n")
        print("1) Buscar por nombre")
        print("2) Buscar por equipo")
        print("3) Buscar por liga")
        print("4) Buscar por posicion")
        print("5) Buscar por pais")
        print("0) Volver")

        opcion = input("\nElige una opcion: ").strip()

        if opcion == "0":
            return presupuesto_restante

        filtros = {}
        if opcion == "1":
            filtros["nombre"] = input("Nombre (completo o parcial): ").strip()
        elif opcion == "2":
            equipos = mostrar_equipos_disponibles(stats_2025)
            equipo = input("Escribe un equipo de la lista o parte del nombre: ").strip()
            if equipo.isdigit() and 1 <= int(equipo) <= len(equipos):
                equipo = equipos[int(equipo) - 1]
            filtros["equipo"] = equipo
        elif opcion == "3":
            filtros["liga"] = input("Liga (AL o NL): ").strip()
        elif opcion == "4":
            filtros["posicion"] = input("Posicion (ej. C, 1B, OF): ").strip()
        elif opcion == "5":
            if "Country" not in stats_2025.columns or stats_2025["Country"].dropna().empty:
                if "PlayerID" in stats_2025.columns:
                    print("\nCargando paises disponibles para la busqueda...")
                    stats_2025 = enriquecer_con_pais(stats_2025)
                if "Country" not in stats_2025.columns or stats_2025["Country"].dropna().empty:
                    print("\nNo hay informacion de pais disponible para estos datos.")
                    input("Presiona Enter para continuar...")
                    continue
            filtros["pais"] = input("Pais (ej. USA, Dominican Republic, Venezuela): ").strip()
        else:
            print("\nOpcion invalida.")
            input("Presiona Enter para continuar...")
            continue

        if not any(filtros.values()):
            print("\nDebes escribir un valor para buscar.")
            input("Presiona Enter para continuar...")
            continue

        resultado = buscar_stats(stats_2025, **filtros).reset_index(drop=True)
        mostrar_resultado_limpio(resultado, con_rango=True)
        presupuesto_restante = ofrecer_draft_desde_resultado(resultado.head(30).reset_index(drop=True), roster, presupuesto_restante, es_pitcher=False)
        input("\nPresiona Enter para volver al menu...")
    return presupuesto_restante


def menu_busqueda(stats_2025, pitchers_2025, bullpen_df, roster, fuente, presupuesto_restante):
    while True:
        mostrar_inicio(roster, presupuesto_restante)
        print("1) Ver leaderboard de hitters")
        print("2) Buscar bateadores y draftear")
        print("3) Ver roster y sugerencias")
        print("4) Buscar equipos por liga y division")
        print("5) Simular temporada y playoffs")
        print("0) Salir")

        opcion = input("\nElige una opcion: ").strip()

        if opcion == "0":
            print("\nSaliendo del buscador.")
            break

        if opcion == "1":
            presupuesto_restante = mostrar_leaderboard_hitters(stats_2025, roster, presupuesto_restante)
            input("\nPresiona Enter para volver al menu...")
            continue
        if opcion == "2":
            presupuesto_restante = menu_busqueda_bateadores(stats_2025, roster, fuente, presupuesto_restante)
            input("\nPresiona Enter para volver al menu...")
            continue
        if opcion == "3":
            mostrar_roster(roster, stats_2025, presupuesto_restante)
            input("\nPresiona Enter para volver al menu...")
            continue
        if opcion == "4":
            mostrar_equipos_disponibles(stats_2025)
            input("\nPresiona Enter para volver al menu...")
            continue
        if opcion == "5":
            simular_temporada_y_playoffs(roster, pitchers_2025, bullpen_df)
            input("\nPresiona Enter para volver al menu...")
            continue

        print("\nOpcion invalida.")
        input("Presiona Enter para continuar...")


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    jugar_franquicia()


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except EOFError:
            print("\n\nEntrada finalizada. Saliendo...")
            break
        except KeyboardInterrupt:
            print("\n\nSe detecto Ctrl+C.")
            try:
                confirmar = input("Deseas salir del programa? (s/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                confirmar = "s"

            if _es_comando_shell(confirmar):
                print("Entrada detectada como comando de terminal. Cerrando el programa.")
                confirmar = "s"

            if confirmar in {"s", "si", "sí", "y", "yes"}:
                print("Saliendo...")
                break
            print("Continuando ejecucion...")