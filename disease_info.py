"""Disease knowledge base for the 38 PlantVillage classes.

Each entry: crop, disease, pathogen, kind, risk, treatment.
`keys` are lowercase fragments matched against the normalised model label.
Risk is the typical agronomic severity of the disease (not the model's confidence).
"""

import re

LOW, MODERATE, HIGH, NONE = "Low Risk", "Moderate Risk", "High Risk", "No Risk"

HEALTHY_TXT = (
    "No visible disease symptoms. Keep up balanced watering and nutrition, "
    "ensure good air flow, and scout weekly so problems are caught early."
)

_E = lambda crop, disease, pathogen, kind, risk, treat, keys: dict(
    crop=crop, disease=disease, pathogen=pathogen, kind=kind, risk=risk, treatment=treat, keys=keys
)
_H = lambda crop: _E(crop, "Healthy", "None detected", "Healthy", NONE, HEALTHY_TXT, ["healthy"])

KB = {
    "apple": [
        _E("Apple", "Apple Scab", "Venturia inaequalis", "Fungal", MODERATE,
           "Olive-green to brown scabby spots on leaves and fruit. Rake and destroy fallen leaves, prune for airflow, "
           "and apply protectant fungicide from green-tip through petal fall in wet springs.", ["scab"]),
        _E("Apple", "Black Rot", "Botryosphaeria obtusa", "Fungal", MODERATE,
           "Frog-eye leaf spots and rotting fruit. Prune out cankers and mummified fruit, "
           "keep trees vigorous, and spray fungicide during bloom to early summer.", ["black rot", "blackrot"]),
        _E("Apple", "Cedar Apple Rust", "Gymnosporangium juniperi-virginianae", "Fungal", MODERATE,
           "Bright orange-yellow leaf spots. Remove nearby juniper/cedar hosts if possible, "
           "plant resistant varieties, and apply fungicide from pink bud through petal fall.", ["cedar", "rust"]),
        _H("Apple"),
    ],
    "blueberry": [_H("Blueberry")],
    "cherry": [
        _E("Cherry", "Powdery Mildew", "Podosphaera clandestina", "Fungal", MODERATE,
           "White powdery growth on leaves and shoots. Prune for light and airflow, avoid excess nitrogen, "
           "and use sulfur or a labelled fungicide at first sign.", ["powdery", "mildew"]),
        _H("Cherry"),
    ],
    "corn": [
        _E("Corn (Maize)", "Gray Leaf Spot", "Cercospora zeae-maydis", "Fungal", MODERATE,
           "Rectangular tan-grey lesions between veins that thrive in humid weather and residue-heavy fields. "
           "Rotate crops, till residue, choose tolerant hybrids, and spray at tasselling if disease is climbing.",
           ["gray leaf", "grey leaf", "cercospora"]),
        _E("Corn (Maize)", "Common Rust", "Puccinia sorghi", "Fungal", LOW,
           "Small brick-red pustules on both leaf surfaces. Usually minor; use resistant hybrids and "
           "spray only if pustules appear early on susceptible varieties.", ["common rust", "rust"]),
        _E("Corn (Maize)", "Northern Leaf Blight", "Exserohilum turcicum", "Fungal", MODERATE,
           "Long cigar-shaped lesions that cost yield when they reach the ear leaf before grain fill. "
           "Plant hybrids carrying Ht resistance genes, rotate out of corn, and treat around tasselling "
           "if lesions are moving up the plant in wet weather.", ["northern", "blight"]),
        _H("Corn (Maize)"),
    ],
    "grape": [
        _E("Grape", "Black Rot", "Guignardia bidwellii", "Fungal", HIGH,
           "Brown leaf spots and shrivelled black fruit mummies. Remove mummies and infected canes, open the canopy, "
           "and apply fungicide from early shoot growth until fruit set.", ["black rot", "blackrot"]),
        _E("Grape", "Esca (Black Measles)", "Phaeomoniella chlamydospora & Phaeoacremonium spp.", "Fungal", HIGH,
           "Tiger-stripe leaf pattern and spotted berries from a wood-rotting complex. No cure: prune in dry weather, "
           "seal large cuts, remove badly affected vines, and use clean planting stock.", ["esca", "measles"]),
        _E("Grape", "Leaf Blight (Isariopsis Leaf Spot)", "Pseudocercospora vitis", "Fungal", MODERATE,
           "Dark irregular leaf spots leading to early defoliation. Improve airflow, clear fallen leaves, "
           "and apply a protectant fungicide if it recurs each season.", ["leaf blight", "isariopsis"]),
        _H("Grape"),
    ],
    "orange": [
        _E("Orange", "Citrus Greening (Huanglongbing)", "Candidatus Liberibacter asiaticus", "Bacterial", HIGH,
           "Blotchy yellow mottling and lopsided bitter fruit, spread by the Asian citrus psyllid. There is no cure: "
           "control psyllids, remove infected trees, and plant certified disease-free nursery stock.",
           ["haunglongbing", "huanglongbing", "greening"]),
    ],
    "peach": [
        _E("Peach", "Bacterial Spot", "Xanthomonas arboricola pv. pruni", "Bacterial", MODERATE,
           "Small angular leaf spots that drop out, leaving a shot-hole look. Choose tolerant varieties, "
           "avoid overhead watering, and apply copper sprays at leaf fall and early spring.", ["bacterial"]),
        _H("Peach"),
    ],
    "pepper": [
        _E("Pepper (Bell)", "Bacterial Spot", "Xanthomonas campestris pv. vesicatoria", "Bacterial", MODERATE,
           "Water-soaked spots turning brown on leaves and raised scabs on fruit. Use clean seed, rotate for 2-3 years, "
           "avoid working wet plants, and apply copper-based bactericide early.", ["bacterial"]),
        _H("Pepper (Bell)"),
    ],
    "potato": [
        _E("Potato", "Early Blight", "Alternaria solani", "Fungal", MODERATE,
           "Dark target-ring spots on older leaves. Rotate crops, keep plants well fed and watered, "
           "and use a protectant fungicide when lesions begin to spread.", ["early blight"]),
        _E("Potato", "Late Blight", "Phytophthora infestans", "Oomycete", HIGH,
           "Fast-spreading water-soaked blotches that can destroy a field in days in cool wet weather. "
           "Remove infected plants immediately, destroy volunteer tubers, and apply late-blight fungicide.", ["late blight"]),
        _H("Potato"),
    ],
    "raspberry": [_H("Raspberry")],
    "soybean": [_H("Soybean")],
    "squash": [
        _E("Squash", "Powdery Mildew", "Podosphaera xanthii", "Fungal", MODERATE,
           "White flour-like patches on leaves reduce yield. Plant resistant varieties with good spacing, "
           "and apply sulfur, potassium bicarbonate or a labelled fungicide early.", ["powdery", "mildew"]),
    ],
    "strawberry": [
        _E("Strawberry", "Leaf Scorch", "Diplocarpon earlianum", "Fungal", MODERATE,
           "Many small purple spots that merge and scorch leaves. Renovate beds after harvest, remove old foliage, "
           "avoid overhead watering and use resistant cultivars.", ["scorch"]),
        _H("Strawberry"),
    ],
    "tomato": [
        _E("Tomato", "Bacterial Spot", "Xanthomonas spp.", "Bacterial", MODERATE,
           "Small dark greasy spots on leaves and rough scabs on fruit. Use clean seed, rotate crops, "
           "avoid overhead irrigation and apply copper-based sprays.", ["bacterial"]),
        _E("Tomato", "Early Blight", "Alternaria solani", "Fungal", MODERATE,
           "Concentric-ring spots on lower leaves. Mulch, stake plants, remove lower infected leaves, "
           "and apply protectant fungicide if it keeps spreading.", ["early blight"]),
        _E("Tomato", "Late Blight", "Phytophthora infestans", "Oomycete", HIGH,
           "Large greasy grey-green blotches and white mould on leaf undersides in cool wet weather. "
           "Remove and bag infected plants at once and apply late-blight fungicide to neighbours.", ["late blight"]),
        _E("Tomato", "Leaf Mold", "Passalora fulva", "Fungal", LOW,
           "Yellow patches above with olive mould beneath, common in humid greenhouses. "
           "Lower humidity, improve ventilation, and remove affected leaves.", ["leaf mold", "leaf mould"]),
        _E("Tomato", "Septoria Leaf Spot", "Septoria lycopersici", "Fungal", MODERATE,
           "Many small round spots with dark borders and grey centres. Remove infected lower leaves, mulch, "
           "rotate crops and apply fungicide when spots first appear.", ["septoria"]),
        _E("Tomato", "Spider Mites", "Tetranychus urticae (two-spotted spider mite)", "Pest", MODERATE,
           "Fine stippling and webbing from tiny sap-sucking mites in hot dry weather. Spray leaf undersides with water, "
           "use insecticidal soap or miticide, and encourage predatory mites.", ["spider", "mite"]),
        _E("Tomato", "Target Spot", "Corynespora cassiicola", "Fungal", MODERATE,
           "Brown spots with concentric rings on leaves and fruit. Improve airflow, avoid wet foliage, "
           "and use a labelled fungicide if it spreads.", ["target"]),
        _E("Tomato", "Yellow Leaf Curl Virus", "Tomato yellow leaf curl virus (TYLCV)", "Viral", HIGH,
           "Upward-curling, yellowed, stunted leaves spread by whiteflies. No cure: remove infected plants, "
           "control whiteflies, use reflective mulch, and plant resistant varieties.", ["yellow leaf curl", "curl"]),
        _E("Tomato", "Mosaic Virus", "Tomato mosaic virus (ToMV)", "Viral", HIGH,
           "Mottled light/dark green leaves and distorted growth, spread by hands and tools. "
           "Remove infected plants, disinfect tools, wash hands, and use resistant varieties.", ["mosaic"]),
        _H("Tomato"),
    ],
}

CROP_ALIASES = {
    "corn": ["corn", "maize"],
    "pepper": ["pepper", "bell"],
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def lookup(label: str) -> dict:
    """Return the knowledge-base entry for a raw model label (fuzzy matched)."""
    n = _norm(label)
    for crop, entries in KB.items():
        names = CROP_ALIASES.get(crop, [crop])
        if not any(name in n.split() for name in names):
            continue
        for e in entries:
            if any(k in n for k in e["keys"]):
                return e
    healthy = "healthy" in n
    return dict(
        crop="Unknown", disease=label.replace("___", " - ").replace("_", " ").strip(),
        pathogen="Not catalogued", kind="Healthy" if healthy else "Unknown",
        risk=NONE if healthy else MODERATE,
        treatment=HEALTHY_TXT if healthy else
        "Isolate the plant, remove affected leaves, and consult a local agricultural extension officer.",
        keys=[],
    )


def display_name(label: str) -> str:
    e = lookup(label)
    return f"{e['crop']} — {e['disease']}" if e["crop"] != "Unknown" else e["disease"]
