# -*- coding: utf-8 -*-
"""
Prototipo de conexion a MusicBrainz API.
Objetivo: confirmar que podemos obtener, para una banda dada:
  - su MBID (identificador unico)
  - sus tags/generos (genero/subgenero declarado por la comunidad)
  - sus relaciones URL (para intentar localizar un enlace a Spotify, usado
    despues para el reproductor via oEmbed)

Requisitos de la API (Politica de uso de MusicBrainz):
  - User-Agent personalizado con nombre de app, version y contacto.
  - Maximo ~1 request/segundo para uso anonimo.
"""
import time
import requests

HEADERS = {
    "User-Agent": "InkriffDataCollector/0.1 (viry.proyecto.diplomado@example.com)"
}
BASE_URL = "https://musicbrainz.org/ws/2"

SAMPLE_BANDS = [
    "Iron Maiden",
    "Nightwish",
    "Opeth",
    "Killswitch Engage",
    "Eluveitie",
]


def search_artist(name):
    url = f"{BASE_URL}/artist"
    params = {"query": f'artist:"{name}"', "fmt": "json", "limit": 1}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("artists"):
        return None
    return data["artists"][0]


def get_artist_detail(mbid):
    url = f"{BASE_URL}/artist/{mbid}"
    params = {"fmt": "json", "inc": "tags+url-rels"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    results = []
    for name in SAMPLE_BANDS:
        artist = search_artist(name)
        time.sleep(1.1)  # respetar rate limit ~1 req/seg
        if not artist:
            print(f"[!] No se encontro: {name}")
            continue

        detail = get_artist_detail(artist["id"])
        time.sleep(1.1)

        tags = sorted(
            detail.get("tags", []), key=lambda t: t.get("count", 0), reverse=True
        )
        tag_names = [t["name"] for t in tags[:5]]

        spotify_url = None
        for rel in detail.get("relations", []):
            url_info = rel.get("url", {})
            resource = url_info.get("resource", "")
            if "spotify.com" in resource:
                spotify_url = resource
                break

        row = {
            "banda": name,
            "mbid": artist["id"],
            "pais": detail.get("country"),
            "tags_top": tag_names,
            "spotify_url": spotify_url,
        }
        results.append(row)
        print(row)

    return results


if __name__ == "__main__":
    main()
