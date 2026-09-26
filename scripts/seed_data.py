# -*- coding: utf-8 -*-
"""
Universo de datos semilla para el proyecto Inkriff.
Lista curada manualmente (no proviene de Kaggle ni de ningun dataset preexistente)
para servir como punto de partida en la extraccion via MusicBrainz, Genius,
Open Library y Google Books.

v2 (expansion): incorpora las ~41 bandas y ~98 libros nuevos que aparecen en
pares_semilla_final.csv (la lista de 105 pares libro<->banda curada por Viry
a partir de su propio conocimiento de fandom). La idea es que el universo de
extraccion cubra tambien los libros/bandas usados como anclas de
entrenamiento supervisado (dual encoder), no solo la muestra original de
2-3 por subgenero/categoria.
"""

# Subgeneros de metal/rock con bandas representativas.
# Las 14 categorias originales se mantienen; se agregan 10 categorias nuevas
# para cubrir subgeneros que no estaban representados (pop punk, post-hardcore,
# alternative metal/rock, gothic rock, occult rock, punk rock, melodic death
# metal, blackened death metal).
SEED_BANDS = {
    # --- categorias originales (expandidas con los pares de Viry) ---
    "Hard Rock / Rock clasico": ["Led Zeppelin", "AC/DC", "Deep Purple"],
    "Heavy Metal": ["Iron Maiden", "Judas Priest", "Black Sabbath", "Avenged Sevenfold"],
    "Power Metal": ["Blind Guardian", "DragonForce", "Sabaton", "Rhapsody of Fire"],
    "Symphonic Metal": ["Nightwish", "Epica", "Within Temptation", "Kamelot"],
    "Folk Metal": ["Eluveitie", "Korpiklaani", "Wintersun"],
    "Gothic Metal": ["Type O Negative", "Paradise Lost", "Lacuna Coil", "Evanescence", "Chelsea Wolfe"],
    "Progressive Metal": ["Dream Theater", "Opeth", "Tool", "Gojira"],
    "Death Metal": ["Death", "Cannibal Corpse", "At the Gates"],
    "Black Metal": ["Mayhem", "Emperor", "Dimmu Borgir"],
    "Metalcore": ["Killswitch Engage", "Bring Me The Horizon", "Architects", "Spiritbox", "Jinjer"],
    "Deathcore": ["Suicide Silence", "Whitechapel", "Lorna Shore"],
    "Nu Metal": ["Slipknot", "Korn", "System of a Down"],
    "Doom Metal / Stoner": ["Electric Wizard", "Sleep", "Candlemass"],
    "Post-Metal / Atmospheric": ["Alcest", "Agalloch"],
    # --- categorias nuevas (a partir de pares_semilla_final.csv) ---
    "Melodic Death Metal": ["Amon Amarth"],
    "Blackened Death Metal": ["Behemoth"],
    "Post-Hardcore": ["Pierce the Veil", "Sleeping With Sirens"],
    "Alternative Metal": ["Sleep Token", "Deftones", "Bad Omens", "Breaking Benjamin", "Disturbed", "Linkin Park"],
    "Emo / Alternative Rock": ["My Chemical Romance"],
    "Pop Punk": ["Fall Out Boy", "The Wonder Years", "Sum 41", "All Time Low", "Paramore",
                 "Blink-182", "Neck Deep", "Simple Plan", "Mayday Parade"],
    "Alternative Rock": ["The Maine", "The 1975", "Florence + The Machine", "The Smashing Pumpkins",
                          "Muse", "The Pretty Reckless", "Royal Blood"],
    "Gothic Rock": ["The Cure", "HIM", "The 69 Eyes"],
    "Occult Rock": ["Ghost"],
    "Punk Rock": ["Green Day"],
}

# Generos/subgeneros de fantasia y romance con libros o series representativos.
# Las 6 categorias originales se mantienen; se agregan 8 categorias nuevas.
SEED_BOOKS = {
    # --- categorias originales (expandidas) ---
    "Alta fantasia / epica": [
        "El Senor de los Anillos",
        "Canción de Hielo y Fuego",
        "El Nombre del Viento",
        "El Hobbit",
        "El Silmarillion",
        "Eragon",
        "El temor de un hombre sabio",
        "Rey de cicatrices",
        "Mistborn: El imperio final",
        "El pozo de la ascensión",
        "El héroe de las eras",
        "El camino de los reyes",
        "El archivo de las tormentas",
        "La rueda del tiempo",
        "El priorato del naranjo",
    ],
    "Romantasy (fantasia romantica)": [
        "Trono de Cristal",
        "Una Corte de Rosas y Espinas",
        "Cuarto Ala",
        "Una corte de niebla y furia",
        "Una corte de alas y ruina",
        "Reina de sombras",
        "Fourth Wing",
        "Iron Flame",
        "Onyx Storm",
        "Powerless",
        "Reckless",
        "Divine Rivals",
        "Ruthless Vows",
        "Érase una vez un corazón roto",
        "La balada del príncipe roto",
        "La vida invisible de Addie LaRue",
        "Stardust",
        "Carry On",
    ],
    "Fantasia oscura / gotica": [
        "Sombra y Hueso",
        "Drácula",
        "Vicious",
        "Berserk",
        "The Witcher",
        "La novena casa",
        "Hell Bent",
        "El océano al final del camino",
        "Elric de Melniboné",
        "The Poppy War",
        "La guerra de la amapola",
        "Coraline",
        "A Darker Shade of Magic",
        "The Locked Tomb",
        "Gideon the Ninth",
        "Mexican Gothic",
        "The Hazel Wood",
        "Jonathan Strange y el señor Norrell",
        "Piranesi",
    ],
    "Romance historico / gotico": [
        "Outlander",
        "Jane Eyre",
        "Un día de diciembre",
    ],
    "Fantasia urbana": [
        "Cazadores de Sombras: Ciudad de Hueso",
        "The Infernal Devices",
        "Cazadores de sombras",
        "Ciudad de hueso",
        "Seis de cuervos",
    ],
    "Mitologia (nordica / celta / griega)": [
        "Mitología Nórdica",  # Neil Gaiman, "Norse Mythology"
        "La canción de Aquiles",
        "Circe",
        "Percy Jackson",
        "El ladrón del rayo",
        "Héroes del Olimpo",
        "Gods of Jade and Shadow",
    ],
    # --- categorias nuevas (a partir de pares_semilla_final.csv) ---
    "Fae Fantasy": [
        "El príncipe cruel",
        "El rey malvado",
        "La reina de nada",
    ],
    "Fantasia clasica / juvenil": [
        "Las crónicas de Narnia",
        "La brújula dorada",
        "El castillo ambulante",
        "Alicia en el País de las Maravillas",
        "A través del espejo",
        "Caraval",
        "El circo de la noche",
        "El mar sin estrellas",
    ],
    "Grimdark": [
        "La primera ley",
        "Malaz: El libro de los caídos",
        "La compañía negra",
    ],
    "Dark Academia": [
        "Babel",
        "Ninth House",
    ],
    "Gotico clasico / vampiros": [
        "Entrevista con el vampiro",
        "El vampiro Lestat",
        "Las brujas de Mayfair",
        "Carmilla",
        "El retrato de Dorian Gray",
        "Frankenstein",
        "El castillo de Otranto",
        "Vampire Academy",
        "A Dowry of Blood",
    ],
    "Distopia juvenil": [
        "Los juegos del hambre",
        "En llamas",
        "Sinsajo",
        "Divergente",
        "Maze Runner",
        "Ready Player One",
        "Scott Pilgrim",
    ],
    "Romance juvenil / coming-of-age": [
        "Las ventajas de ser invisible",
        "Heartstopper",
        "A todos los chicos de los que me enamoré",
        "Eleanor & Park",
    ],
    "Fantasia de cuento de hadas": [
        "Nettle & Bone",
        "Uprooted",
        "Spinning Silver",
        "A Deadly Education",
        "The Bear and the Nightingale",
        "The Once and Future Witches",
        "The Invisible Life of Addie LaRue",
    ],
    "Cozy Fantasy / sobrenatural": [
        "La casa en el mar más azul",
        "Under the Whispering Door",
    ],
}

if __name__ == "__main__":
    n_bands = sum(len(v) for v in SEED_BANDS.values())
    n_books = sum(len(v) for v in SEED_BOOKS.values())
    print(f"Subgeneros de metal/rock: {len(SEED_BANDS)} | bandas semilla: {n_bands}")
    print(f"Generos de fantasia/romance: {len(SEED_BOOKS)} | libros semilla: {n_books}")

    # Verificacion: todas las bandas/libros de pares_semilla_final.csv deben
    # estar cubiertos por el universo de extraccion (con tolerancia a variantes
    # de mayusculas/acentos entre el titulo "canonico" de la semilla y el
    # titulo usado en el par, que puede venir en ingles o con autor).
    try:
        from build_pares_final import PARES
        bandas_semilla = {b for lst in SEED_BANDS.values() for b in lst}
        libros_semilla = {b for lst in SEED_BOOKS.values() for b in lst}
        bandas_pares = {p[2].split(" — ")[0] for p in PARES}
        faltan_bandas = sorted(bandas_pares - bandas_semilla)
        if faltan_bandas:
            print(f"AVISO: bandas en pares_semilla_final.csv sin match exacto en SEED_BANDS: {faltan_bandas}")
        else:
            print("OK: todas las bandas de pares_semilla_final.csv estan en SEED_BANDS.")

        libros_pares = {p[0].split(" — ")[0] for p in PARES}
        faltan_libros = sorted(libros_pares - libros_semilla)
        if faltan_libros:
            print(f"AVISO: libros en pares_semilla_final.csv sin match exacto en SEED_BOOKS: {faltan_libros}")
        else:
            print("OK: todos los libros de pares_semilla_final.csv estan en SEED_BOOKS.")
    except ImportError:
        pass
