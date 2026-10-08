"""Disease knowledge base for the 38 PlantVillage classes.

Each entry: crop, disease, pathogen, kind, risk, treatment, prevention, organic.
`keys` are lowercase fragments matched against the normalised model label.
Risk is the typical agronomic severity of the disease (not the model's confidence).
"""

import re

LOW, MODERATE, HIGH, NONE = "Low Risk", "Moderate Risk", "High Risk", "No Risk"

HEALTHY_TXT = (
    "No visible disease symptoms. Keep up balanced watering and nutrition, "
    "ensure good air flow, and scout weekly so problems are caught early."
)
HEALTHY_PREV = [
    "Scout weekly for early signs of disease",
    "Maintain balanced fertilisation — avoid excess nitrogen",
    "Ensure good air circulation through proper spacing",
    "Water at the base; keep foliage dry",
]
HEALTHY_ORG = [
    "Compost tea foliar spray to boost leaf microbiome",
    "Neem oil bi-weekly as a preventive",
    "Mulch around the base to reduce soil splash",
]

_E = lambda crop, disease, pathogen, kind, risk, treat, prev, org, keys: dict(
    crop=crop, disease=disease, pathogen=pathogen, kind=kind, risk=risk,
    treatment=treat, prevention=prev, organic=org, keys=keys
)
_H = lambda crop: _E(
    crop, "Healthy", "None detected", "Healthy", NONE,
    HEALTHY_TXT, HEALTHY_PREV, HEALTHY_ORG, ["healthy"]
)

KB = {
    "apple": [
        _E("Apple", "Apple Scab", "Venturia inaequalis", "Fungal", MODERATE,
           "Olive-green to brown scabby spots on leaves and fruit. Rake and destroy fallen leaves, prune for airflow, "
           "and apply protectant fungicide from green-tip through petal fall in wet springs.",
           [
               "Plant scab-resistant varieties (e.g. Liberty, Enterprise)",
               "Rake and destroy fallen leaves every autumn",
               "Prune to open canopy and improve air flow",
               "Apply preventive fungicide from green-tip through petal fall in wet seasons",
           ],
           [
               "Sulfur spray from bud-break (do not apply within 2 weeks of oil)",
               "Copper hydroxide at early green-tip stage",
               "Bacillus subtilis (Serenade) as a biocontrol foliar spray",
               "Neem oil on a 7-day schedule during primary infection period",
           ],
           ["scab"]),
        _E("Apple", "Black Rot", "Botryosphaeria obtusa", "Fungal", MODERATE,
           "Frog-eye leaf spots and rotting fruit. Prune out cankers and mummified fruit, "
           "keep trees vigorous, and spray fungicide during bloom to early summer.",
           [
               "Remove and destroy mummified fruit and cankered wood each winter",
               "Disinfect pruning tools between cuts with 70% ethanol",
               "Keep trees vigorous with proper nutrition; stressed trees are more susceptible",
               "Apply fungicide preventively during bloom in wet years",
           ],
           [
               "Copper-based fungicide applied at bloom",
               "Lime sulfur during dormant season on cankers",
               "Bacillus subtilis biocontrol spray during the season",
               "Ensure good drainage; water-logged soil stresses trees",
           ],
           ["black rot", "blackrot"]),
        _E("Apple", "Cedar Apple Rust", "Gymnosporangium juniperi-virginianae", "Fungal", MODERATE,
           "Bright orange-yellow leaf spots. Remove nearby juniper/cedar hosts if possible, "
           "plant resistant varieties, and apply fungicide from pink bud through petal fall.",
           [
               "Remove nearby Eastern red cedar or juniper trees within 1–2 miles if feasible",
               "Plant rust-resistant apple varieties (Liberty, Redfree, Pristine)",
               "Apply protective fungicide from pink bud through petal fall",
               "Inspect junipers for orange galls in spring and remove them",
           ],
           [
               "Sulfur fungicide from pink bud through petal fall (7–10 day intervals)",
               "Copper spray at early green-tip stage",
               "Neem oil applied before visible infection",
               "Kaolin clay particle film as a physical barrier",
           ],
           ["cedar", "rust"]),
        _H("Apple"),
    ],
    "blueberry": [_H("Blueberry")],
    "cherry": [
        _E("Cherry", "Powdery Mildew", "Podosphaera clandestina", "Fungal", MODERATE,
           "White powdery growth on leaves and shoots. Prune for light and airflow, avoid excess nitrogen, "
           "and use sulfur or a labelled fungicide at first sign.",
           [
               "Select resistant or tolerant cherry varieties",
               "Prune annually for open canopy and good sunlight penetration",
               "Avoid excess nitrogen fertiliser — lush growth is more susceptible",
               "Apply preventive sulfur spray at bud-break in seasons with history of mildew",
           ],
           [
               "Sulfur dust or wettable sulfur at first sign (avoid in heat >32 °C)",
               "Potassium bicarbonate spray — alters leaf surface pH against fungus",
               "Neem oil (azadirachtin) on a 7-day rotation",
               "Bacillus subtilis (Serenade) foliar spray as biocontrol",
           ],
           ["powdery", "mildew"]),
        _H("Cherry"),
    ],
    "corn": [
        _E("Corn (Maize)", "Gray Leaf Spot", "Cercospora zeae-maydis", "Fungal", MODERATE,
           "Rectangular tan-grey lesions between veins that thrive in humid weather and residue-heavy fields. "
           "Rotate crops, till residue, choose tolerant hybrids, and spray at tasselling if disease is climbing.",
           [
               "Rotate maize with non-host crops (soybean, wheat) for at least one season",
               "Choose hybrids with high GLS tolerance ratings",
               "Till crop residue post-harvest to reduce inoculum",
               "Avoid planting in low-lying fields with poor air drainage",
           ],
           [
               "Copper fungicide spray at early tasselling if conditions are humid",
               "Trichoderma-based soil inoculant to improve root health and resilience",
               "Avoid excessive nitrogen; balanced K nutrition improves resistance",
               "Biocontrol Bacillus amyloliquefaciens seed treatment",
           ],
           ["gray leaf", "grey leaf", "cercospora"]),
        _E("Corn (Maize)", "Common Rust", "Puccinia sorghi", "Fungal", LOW,
           "Small brick-red pustules on both leaf surfaces. Usually minor; use resistant hybrids and "
           "spray only if pustules appear early on susceptible varieties.",
           [
               "Plant Rp-gene resistant hybrids where rust is historically severe",
               "Early planting to outgrow peak rust pressure in mid-season",
               "Scout from V6 onward; low early-season levels rarely warrant action",
               "Avoid continuous maize in fields with high rust history",
           ],
           [
               "Sulfur spray if pustules appear before V8 on a susceptible hybrid",
               "Neem oil foliar application at early pustule detection",
               "Potassium silicate foliar spray to strengthen leaf epidermis",
               "Balanced nutrition — avoid excess N which increases susceptibility",
           ],
           ["common rust", "rust"]),
        _E("Corn (Maize)", "Northern Leaf Blight", "Exserohilum turcicum", "Fungal", MODERATE,
           "Long cigar-shaped lesions that cost yield when they reach the ear leaf before grain fill. "
           "Plant hybrids carrying Ht resistance genes, rotate out of corn, and treat around tasselling "
           "if lesions are moving up the plant in wet weather.",
           [
               "Use hybrids with Ht1, Ht2, or HtN resistance genes",
               "Rotate with non-host crops for 1–2 seasons",
               "Reduce residue through tillage or cover cropping",
               "Avoid dense plant populations that reduce airflow",
           ],
           [
               "Copper-based fungicide at tasselling in warm, wet conditions",
               "Bacillus subtilis foliar spray as a biocontrol preventive",
               "Sulfur spray when lesions first appear on lower leaves",
               "Avoid overhead irrigation; use drip irrigation if possible",
           ],
           ["northern", "blight"]),
        _H("Corn (Maize)"),
    ],
    "grape": [
        _E("Grape", "Black Rot", "Guignardia bidwellii", "Fungal", HIGH,
           "Brown leaf spots and shrivelled black fruit mummies. Remove mummies and infected canes, open the canopy, "
           "and apply fungicide from early shoot growth until fruit set.",
           [
               "Remove and destroy all mummified fruit and infected canes each winter",
               "Train vines to maximise sunlight and air penetration",
               "Begin preventive fungicide programme at early shoot growth (< 5 cm)",
               "Scout closely during bloom — infection during this stage is most damaging",
           ],
           [
               "Sulfur fungicide from early shoot growth until post-bloom (7-day intervals in wet weather)",
               "Copper hydroxide spray at early growth stages",
               "Bacillus subtilis biocontrol applied at bloom and fruit set",
               "Kaolin clay film on developing berries as a physical barrier",
           ],
           ["black rot", "blackrot"]),
        _E("Grape", "Esca (Black Measles)", "Phaeomoniella chlamydospora & Phaeoacremonium spp.", "Fungal", HIGH,
           "Tiger-stripe leaf pattern and spotted berries from a wood-rotting complex. No cure: prune in dry weather, "
           "seal large cuts, remove badly affected vines, and use clean planting stock.",
           [
               "Purchase certified disease-free planting material",
               "Prune only in dry weather; avoid pruning during rain or high humidity",
               "Seal pruning wounds > 2 cm immediately with wound sealant",
               "Do not replant removed vines with the same variety in the same site",
           ],
           [
               "Trichoderma atroviride applied to pruning wounds as biocontrol",
               "Wound paste containing Trichoderma or copper powder",
               "Avoid water stress — steady drip irrigation reduces plant stress",
               "There is no curative treatment; focus entirely on prevention",
           ],
           ["esca", "measles"]),
        _E("Grape", "Leaf Blight (Isariopsis Leaf Spot)", "Pseudocercospora vitis", "Fungal", MODERATE,
           "Dark irregular leaf spots leading to early defoliation. Improve airflow, clear fallen leaves, "
           "and apply a protectant fungicide if it recurs each season.",
           [
               "Clear fallen leaves from beneath vines to reduce overwintering inoculum",
               "Improve canopy ventilation by leaf removal on the fruit zone",
               "Avoid overhead irrigation; use drip watering",
               "Apply preventive copper spray after heavy rain events",
           ],
           [
               "Copper oxychloride spray at early sign of spotting",
               "Neem oil foliar treatment every 10–14 days during wet periods",
               "Potassium bicarbonate spray as a surface treatment",
               "Bacillus subtilis (Serenade) as a biocontrol option",
           ],
           ["leaf blight", "isariopsis"]),
        _H("Grape"),
    ],
    "orange": [
        _E("Orange", "Citrus Greening (Huanglongbing)", "Candidatus Liberibacter asiaticus", "Bacterial", HIGH,
           "Blotchy yellow mottling and lopsided bitter fruit, spread by the Asian citrus psyllid. There is no cure: "
           "control psyllids, remove infected trees, and plant certified disease-free nursery stock.",
           [
               "Plant certified psyllid-free nursery stock only",
               "Install fine mesh screens around nurseries",
               "Monitor weekly for the Asian citrus psyllid (ACP) — spray at first detection",
               "Remove and destroy infected trees immediately to reduce spread",
           ],
           [
               "Kaolin clay particle film repels Asian citrus psyllid adults",
               "Insecticidal soap or neem oil to suppress psyllid populations",
               "Lacewing or parasitic wasp releases (Tamarixia radiata) as biological control",
               "There is no organic cure — prevention through psyllid control is essential",
           ],
           ["haunglongbing", "huanglongbing", "greening"]),
    ],
    "peach": [
        _E("Peach", "Bacterial Spot", "Xanthomonas arboricola pv. pruni", "Bacterial", MODERATE,
           "Small angular leaf spots that drop out, leaving a shot-hole look. Choose tolerant varieties, "
           "avoid overhead watering, and apply copper sprays at leaf fall and early spring.",
           [
               "Plant tolerant peach varieties (e.g. Contender, Reliance, Redhaven)",
               "Avoid overhead irrigation; use drip systems",
               "Apply copper bactericide at leaf fall and during dormant season",
               "Avoid planting in low areas prone to late spring frosts (frost damage opens entry points)",
           ],
           [
               "Copper hydroxide spray at leaf fall and green tip (copper resistance is a risk — rotate)",
               "Copper octanoate (soap-based copper) — gentler on foliage",
               "Avoid high nitrogen fertilisation — lush shoots are more susceptible",
               "Kaolin clay particle film as a physical barrier during fruit development",
           ],
           ["bacterial"]),
        _H("Peach"),
    ],
    "pepper": [
        _E("Pepper (Bell)", "Bacterial Spot", "Xanthomonas campestris pv. vesicatoria", "Bacterial", MODERATE,
           "Water-soaked spots turning brown on leaves and raised scabs on fruit. Use clean seed, rotate for 2-3 years, "
           "avoid working wet plants, and apply copper-based bactericide early.",
           [
               "Use certified pathogen-free seed or hot-water treat seed at 50 °C for 25 min",
               "Rotate peppers with non-solanaceous crops for 2–3 years",
               "Avoid working in the field when plants are wet",
               "Use drip irrigation and plastic mulch to minimise soil splash",
           ],
           [
               "Copper hydroxide preventive spray starting before first symptoms",
               "Bacillus subtilis biocontrol spray on a 5–7 day schedule",
               "Acibenzolar-S-methyl (plant activator) to trigger systemic resistance",
               "Remove and destroy heavily infected leaves promptly",
           ],
           ["bacterial"]),
        _H("Pepper (Bell)"),
    ],
    "potato": [
        _E("Potato", "Early Blight", "Alternaria solani", "Fungal", MODERATE,
           "Dark target-ring spots on older leaves. Rotate crops, keep plants well fed and watered, "
           "and use a protectant fungicide when lesions begin to spread.",
           [
               "Rotate potato with non-solanaceous crops for 2–3 years",
               "Use certified disease-free seed potatoes",
               "Maintain adequate potassium and calcium nutrition",
               "Scout regularly; begin fungicide programme before lesions exceed 5% foliage",
           ],
           [
               "Copper-based fungicide applied at first symptom and every 7–10 days in wet weather",
               "Bacillus subtilis (Serenade) spray as biocontrol",
               "Neem oil spray on 7-day intervals at early infection",
               "Remove and compost infected lower leaves promptly",
           ],
           ["early blight"]),
        _E("Potato", "Late Blight", "Phytophthora infestans", "Oomycete", HIGH,
           "Fast-spreading water-soaked blotches that can destroy a field in days in cool wet weather. "
           "Remove infected plants immediately, destroy volunteer tubers, and apply late-blight fungicide.",
           [
               "Plant certified blight-free seed potatoes; destroy all volunteer tubers",
               "Choose blight-resistant varieties (e.g. Sarpo Mira, Bionica)",
               "Begin preventive fungicide programme when weather is cool (10–20 °C) and wet",
               "Earth up rows to protect tubers; destroy haulm (foliage) before harvest",
           ],
           [
               "Copper hydroxide or Bordeaux mixture applied on a 5–7 day preventive schedule",
               "Potassium phosphonate (phosphorous acid) — systemic action, acceptable in many organic systems",
               "Remove and bag infected plants immediately — do not compost",
               "Avoid overhead irrigation; water early morning so foliage dries quickly",
           ],
           ["late blight"]),
        _H("Potato"),
    ],
    "raspberry": [_H("Raspberry")],
    "soybean": [_H("Soybean")],
    "squash": [
        _E("Squash", "Powdery Mildew", "Podosphaera xanthii", "Fungal", MODERATE,
           "White flour-like patches on leaves reduce yield. Plant resistant varieties with good spacing, "
           "and apply sulfur, potassium bicarbonate or a labelled fungicide early.",
           [
               "Choose mildew-resistant squash/zucchini varieties",
               "Space plants to maximise airflow between leaves",
               "Avoid shaded or humid planting spots",
               "Apply preventive potassium bicarbonate spray at the start of warm, dry seasons",
           ],
           [
               "Potassium bicarbonate — disrupts fungal cell membranes; spray at first sign",
               "Baking soda (1 tsp/L + horticultural oil) — affordable home spray",
               "Neem oil every 7 days during peak mildew weather",
               "Sulfur dust or wettable sulfur (avoid above 32 °C)",
           ],
           ["powdery", "mildew"]),
    ],
    "strawberry": [
        _E("Strawberry", "Leaf Scorch", "Diplocarpon earlianum", "Fungal", MODERATE,
           "Many small purple spots that merge and scorch leaves. Renovate beds after harvest, remove old foliage, "
           "avoid overhead watering and use resistant cultivars.",
           [
               "Select resistant varieties (e.g. Allstar, Surecrop)",
               "Renovate beds after harvest — mow, thin, remove old leaves",
               "Use drip irrigation to keep foliage dry",
               "Apply preventive copper spray at early spring flush",
           ],
           [
               "Copper-based fungicide spray at early spring and after renovation",
               "Neem oil on a 10-day schedule during wet periods",
               "Remove and dispose of infected leaves promptly",
               "Bacillus subtilis spray as a biocontrol option at first symptom",
           ],
           ["scorch"]),
        _H("Strawberry"),
    ],
    "tomato": [
        _E("Tomato", "Bacterial Spot", "Xanthomonas spp.", "Bacterial", MODERATE,
           "Small dark greasy spots on leaves and rough scabs on fruit. Use clean seed, rotate crops, "
           "avoid overhead irrigation and apply copper-based sprays.",
           [
               "Use certified disease-free seed; hot-water treat seed at 50 °C for 25 min",
               "Rotate tomato with non-solanaceous crops for 2–3 years",
               "Use drip irrigation; avoid working in field when plants are wet",
               "Apply copper bactericide from transplant stage in a preventive programme",
           ],
           [
               "Copper hydroxide spray every 5–7 days in warm, wet weather",
               "Bacillus subtilis (Serenade) as biocontrol on a 5-day schedule",
               "Acibenzolar-S-methyl plant activator to induce SAR resistance",
               "Remove and bag heavily infected leaves and stems promptly",
           ],
           ["bacterial"]),
        _E("Tomato", "Early Blight", "Alternaria solani", "Fungal", MODERATE,
           "Concentric-ring spots on lower leaves. Mulch, stake plants, remove lower infected leaves, "
           "and apply protectant fungicide if it keeps spreading.",
           [
               "Stake or cage plants to keep foliage off the ground",
               "Apply organic mulch to reduce soil splash onto lower leaves",
               "Remove and destroy infected lower leaves at first sign",
               "Rotate tomatoes out of the bed for 2–3 years",
           ],
           [
               "Copper fungicide spray every 7–10 days once spots appear",
               "Bacillus subtilis biocontrol spray starting at transplanting",
               "Neem oil (azadirachtin) on a 7-day preventive schedule",
               "Sulfur spray on lower leaves in humid conditions",
           ],
           ["early blight"]),
        _E("Tomato", "Late Blight", "Phytophthora infestans", "Oomycete", HIGH,
           "Large greasy grey-green blotches and white mould on leaf undersides in cool wet weather. "
           "Remove and bag infected plants at once and apply late-blight fungicide to neighbours.",
           [
               "Plant resistant varieties (e.g. Legend, Mountain Magic, Defiant)",
               "Begin copper spray programme before cool-wet weather arrives",
               "Stake and prune for airflow; avoid overhead watering",
               "Destroy volunteer tomato and potato plants nearby — they harbour the pathogen",
           ],
           [
               "Copper hydroxide or Bordeaux mixture applied preventively every 5–7 days",
               "Potassium phosphonate (phosphorous acid) — systemic, widely allowed in organic systems",
               "Remove infected plants immediately; seal in plastic bags before disposal",
               "Avoid composting infected material — blight pathogen persists",
           ],
           ["late blight"]),
        _E("Tomato", "Leaf Mold", "Passalora fulva", "Fungal", LOW,
           "Yellow patches above with olive mould beneath, common in humid greenhouses. "
           "Lower humidity, improve ventilation, and remove affected leaves.",
           [
               "Keep greenhouse humidity below 85% using vents and fans",
               "Space plants widely; remove lower leaves to improve airflow",
               "Use drip irrigation; avoid wetting leaves",
               "Choose mold-resistant tomato varieties for greenhouse growing",
           ],
           [
               "Potassium bicarbonate spray on affected foliage",
               "Copper-based fungicide if spread is rapid",
               "Bacillus subtilis (Serenade) biocontrol on a 7-day schedule",
               "Remove and dispose of affected leaves promptly",
           ],
           ["leaf mold", "leaf mould"]),
        _E("Tomato", "Septoria Leaf Spot", "Septoria lycopersici", "Fungal", MODERATE,
           "Many small round spots with dark borders and grey centres. Remove infected lower leaves, mulch, "
           "rotate crops and apply fungicide when spots first appear.",
           [
               "Rotate tomato with non-solanaceous crops for 2+ years",
               "Mulch heavily to prevent soil splash onto lower leaves",
               "Remove infected lower leaves as soon as spots appear",
               "Stake plants to keep foliage off the ground",
           ],
           [
               "Copper fungicide spray every 7–10 days once lesions detected",
               "Neem oil as a preventive and curative spray on 7-day intervals",
               "Bacillus subtilis (Serenade) biocontrol — apply at first sign",
               "Remove infected foliage and dispose outside the garden area",
           ],
           ["septoria"]),
        _E("Tomato", "Spider Mites", "Tetranychus urticae (two-spotted spider mite)", "Pest", MODERATE,
           "Fine stippling and webbing from tiny sap-sucking mites in hot dry weather. Spray leaf undersides with water, "
           "use insecticidal soap or miticide, and encourage predatory mites.",
           [
               "Maintain adequate irrigation — drought-stressed plants attract mites",
               "Encourage natural enemies: predatory mites (Phytoseiidae), lacewings",
               "Avoid broad-spectrum insecticides that kill mite predators",
               "Scout leaf undersides weekly in hot, dry periods",
           ],
           [
               "Strong water spray on leaf undersides to dislodge mites",
               "Insecticidal soap spray every 3–5 days until mites are controlled",
               "Neem oil (azadirachtin) disrupts mite reproduction",
               "Release predatory mite Phytoseiulus persimilis for biocontrol",
           ],
           ["spider", "mite"]),
        _E("Tomato", "Target Spot", "Corynespora cassiicola", "Fungal", MODERATE,
           "Brown spots with concentric rings on leaves and fruit. Improve airflow, avoid wet foliage, "
           "and use a labelled fungicide if it spreads.",
           [
               "Prune lower leaves and stake plants for airflow",
               "Use drip irrigation; avoid wetting foliage",
               "Rotate with non-solanaceous crops",
               "Begin preventive spray programme in warm, wet weather",
           ],
           [
               "Copper-based fungicide applied at first symptom on 7-day schedule",
               "Bacillus subtilis biocontrol spray preventively",
               "Neem oil foliar spray every 7–10 days",
               "Remove and dispose of infected leaves and fruit",
           ],
           ["target"]),
        _E("Tomato", "Yellow Leaf Curl Virus", "Tomato yellow leaf curl virus (TYLCV)", "Viral", HIGH,
           "Upward-curling, yellowed, stunted leaves spread by whiteflies. No cure: remove infected plants, "
           "control whiteflies, use reflective mulch, and plant resistant varieties.",
           [
               "Plant TYLCV-resistant tomato varieties (e.g. Shanty, Syngenta HM series)",
               "Use UV-reflective silver mulch to repel whitefly adults",
               "Cover transplants with insect-proof mesh for first 3–4 weeks",
               "Remove and destroy infected plants immediately to reduce spread",
           ],
           [
               "Yellow sticky traps to monitor and mass-capture whiteflies",
               "Insecticidal soap spray on leaf undersides to suppress whitefly populations",
               "Neem oil (azadirachtin) as a whitefly repellent/contact spray",
               "Encarsia formosa parasitic wasp releases for biocontrol in enclosed spaces",
           ],
           ["yellow leaf curl", "curl"]),
        _E("Tomato", "Mosaic Virus", "Tomato mosaic virus (ToMV)", "Viral", HIGH,
           "Mottled light/dark green leaves and distorted growth, spread by hands and tools. "
           "Remove infected plants, disinfect tools, wash hands, and use resistant varieties.",
           [
               "Use ToMV-resistant tomato varieties (look for Tm-2²  gene on the label)",
               "Wash hands with soap before and after handling plants",
               "Disinfect all tools with 10% bleach or 70% alcohol between cuts",
               "Do not smoke near plants — tobacco carries Tobacco mosaic virus (related)",
           ],
           [
               "There is no curative organic treatment for viral disease",
               "Remove and bag infected plants; do not compost",
               "Control aphids with insecticidal soap to reduce secondary virus spread",
               "Apply milk spray (10% skim milk) — may inactivate virus particles on leaf surfaces",
           ],
           ["mosaic"]),
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
        prevention=["Scout regularly", "Ensure good airflow", "Use disease-free planting material"],
        organic=["Neem oil spray", "Copper-based spray if fungal", "Consult local organic extension services"],
        keys=[],
    )


def display_name(label: str) -> str:
    e = lookup(label)
    return f"{e['crop']} — {e['disease']}" if e["crop"] != "Unknown" else e["disease"]
