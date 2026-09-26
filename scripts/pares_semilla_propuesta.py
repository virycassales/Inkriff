# -*- coding: utf-8 -*-
"""Propuesta inicial de pares semilla libro<->banda/subgenero para anclar el dual
encoder supervisado. Es un DRAFT para que Viry lo corrija con su criterio de fandom -
las conexiones libro-banda son subjetivas por naturaleza; aqui se documenta el porque
de cada propuesta para que sea facil de evaluar y ajustar."""
import csv

PARES = [
    {
        "libro": "El Senor de los Anillos",
        "categoria_libro": "Alta fantasia / epica",
        "banda_subgenero": "Blind Guardian (Power Metal)",
        "razon": "Blind Guardian tiene un album entero (Nightfall in Middle-Earth) basado en el Silmarillion de Tolkien; conexion directa y muy conocida en el fandom.",
    },
    {
        "libro": "El Senor de los Anillos",
        "categoria_libro": "Alta fantasia / epica",
        "banda_subgenero": "Eluveitie (Folk Metal)",
        "razon": "Alternativa: instrumentacion folk/celta evoca el tono pastoral de la Comarca y los pueblos de la Tierra Media (pareo secundario, no exclusivo).",
    },
    {
        "libro": "Cancion de Hielo y Fuego",
        "categoria_libro": "Alta fantasia / epica",
        "banda_subgenero": "Type O Negative (Gothic Metal)",
        "razon": "Tono sombrio, tragico y moralmente ambiguo; atmosfera pesada y dramatica que combina con la brutalidad politica de la saga.",
    },
    {
        "libro": "El Nombre del Viento",
        "categoria_libro": "Alta fantasia / epica",
        "banda_subgenero": "Opeth (Progressive Metal)",
        "razon": "El protagonista es musico y narrador; Opeth combina composicion intrincada con pasajes melodicos/acusticos, similar a la mezcla de accion y lirismo del libro.",
    },
    {
        "libro": "Trono de Cristal",
        "categoria_libro": "Romantasy (fantasia romantica)",
        "banda_subgenero": "Killswitch Engage (Metalcore)",
        "razon": "Protagonista asesina, ritmo de accion constante; energia agresiva pero melodica del metalcore combina con el tono de aventura/combate.",
    },
    {
        "libro": "Una Corte de Rosas y Espinas",
        "categoria_libro": "Romantasy (fantasia romantica)",
        "banda_subgenero": "Within Temptation (Symphonic Metal)",
        "razon": "Belleza y peligro entrelazados, mundo feerico, romance dramatico; el symphonic metal aporta esa grandiosidad orquestal y romantica.",
    },
    {
        "libro": "Cuarto Ala",
        "categoria_libro": "Romantasy (fantasia romantica)",
        "banda_subgenero": "Bring Me the Horizon (Metalcore)",
        "razon": "Adrenalina constante (dragones, guerra, enemies-to-lovers); energia intensa y contemporanea del metalcore moderno.",
    },
    {
        "libro": "Sombra y Hueso",
        "categoria_libro": "Fantasia oscura / gotica",
        "banda_subgenero": "Lacuna Coil (Gothic Metal)",
        "razon": "Dualidad luz/sombra, atractivo oscuro del antagonista (el Darkling); el gothic metal vive exactamente en esa tension romantico-oscura.",
    },
    {
        "libro": "Dracula",
        "categoria_libro": "Fantasia oscura / gotica",
        "banda_subgenero": "Paradise Lost (Gothic Metal)",
        "razon": "Referencia directa del genero gothic metal (nacio ligado a esta estetica victoriana/vampirica); pareo casi canonico.",
    },
    {
        "libro": "Vicious",
        "categoria_libro": "Fantasia oscura / gotica",
        "banda_subgenero": "Death (Death Metal)",
        "razon": "Antiheroes sin filtro moral, rivalidad brutal; la crudeza del death metal combina con la falta de heroes claros en la novela.",
    },
    {
        "libro": "Outlander",
        "categoria_libro": "Romance historico / gotico",
        "banda_subgenero": "Eluveitie (Folk Metal)",
        "razon": "Ambientacion en las Highlands escocesas; el folk metal celta encaja directamente con el escenario historico/cultural del libro.",
    },
    {
        "libro": "Jane Eyre",
        "categoria_libro": "Romance historico / gotico",
        "banda_subgenero": "Paradise Lost (Gothic Metal)",
        "razon": "Romance gotico clasico britanico; atmosfera contenida y oscura que es la raiz historica del genero gothic metal.",
    },
    {
        "libro": "Cazadores de Sombras: Ciudad de Hueso",
        "categoria_libro": "Fantasia urbana",
        "banda_subgenero": "System of a Down (Nu Metal)",
        "razon": "Fantasia urbana contemporanea en Nueva York, energia moderna y agresiva; el nu metal tiene esa misma estetica urbana/actual.",
    },
    {
        "libro": "Norse Mythology",
        "categoria_libro": "Mitologia (nordica / celta)",
        "banda_subgenero": "Wintersun (Folk Metal)",
        "razon": "Reescritura directa de mitologia nordica; Wintersun trabaja explicitamente temas nordicos/naturales y escala epica (Ragnarok, dioses).",
    },
]

with open("pares_semilla_propuesta.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["libro", "categoria_libro", "banda_subgenero", "razon"])
    writer.writeheader()
    writer.writerows(PARES)

print(f"{len(PARES)} pares propuestos -> pares_semilla_propuesta.csv")
