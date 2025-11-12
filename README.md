# NNProj - Detekcija AI Generiranih Slika

Projekt za predmet "Neuronske Mreže" - Prepoznavanje umjetno (AI) generiranih slika

## O projektu

Ovaj projekt implementira i uspoređuje različite modele strojnog učenja za detekciju umjetno generiranih slika. Cilj je razlikovati prave slike od umjetno generiranih koristeći različite pristupe i arhitekture.

## Struktura projekta

```
NNProj/
├── src/
│   ├── venv/           # Virtualno okruženje
│   ├── models/         # Implementacije modela
│   ├── data/           # Direktorij za podatke
│   └── utils/          # Pomoćne funkcije
├── notebooks/          # Jupyter bilježnice
├── requirements.txt    # Python ovisnosti
├── setup.sh           # Skripta za automatsku instalaciju
├── clean.sh           # Skripta za čišćenje projekta
└── README.md          # Dokumentacija
```

## Instalacija

### Automatska instalacija (preporučeno)

```bash
git clone https://github.com/FrenkieL/NNProj.git
cd NNProj
chmod +x setup.sh
./setup.sh
```

Skripta `setup.sh` automatski:
- Kreira virtualno okruženje u `src/venv/`
- Instalira sve potrebne pakete iz `requirements.txt`
- Postavlja projekt za rad

### Aktivacija virtualnog okruženja

```bash
# Za Bash/Zsh:
source src/venv/bin/activate

# Za Fish:
source src/venv/bin/activate.fish
```

### Ručna instalacija

Ako preferirate ručno postavljanje bez `setup.sh`:

Sa virtualnim okruženjem:
```bash
python -m venv src/venv
source src/venv/bin/activate
pip install -r requirements.txt
```

Bez virtualnog okruženja (**pip će instalirati pakete globalno**):
```bash
pip install -r requirements.txt
```

## Čišćenje projekta

Za brisanje virtualnog okruženja i privremenih datoteka:

```bash
chmod +x clean.sh
./clean.sh
```

## Autori

- **FrenkieL** - [GitHub](https://github.com/FrenkieL)
- **Gl1tc6** - [GitHub](https://github.com/Gl1tc6)

## Licenca

Projekt kreiran u edukacijske svrhe kao dio kolegija [**Neuronske mreže**](https://www.fer.unizg.hr/predmet/neumre) na [**FER-u**](https://www.fer.unizg.hr/).
