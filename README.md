# Voorspellen data Google Play Store Apps dataset

## Nieuws
- [24/02/2023]: Feature engineering klaar!!
- [20/03/2023]: Repo gemaakt, yay

## Introductie
Data Science opdracht voor Smart Data People. Hier wordt gebruikt gemaakt van de [Google Play Store Apps dataset](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps). 

## Opstarten

**Let op:** Nog niet af, dus kan nog veel veranderen

1. Kloon de repo

2. Installeer de dependencies:

maak een (conda) environment aan speciaal voor dit project.
In VS Code open je de cloned repo, dit is je workspace. Ook open je een nieuwe command prompt (powershell werkt niet standaard, vermoedelijk wanneer je t.t.v. installatie conda niet aan PATH hebt toegevoegd) in terminal.
Maak de environment in de repo, naam kun je evt. aanpassen in het prefix argument:
conda env create --file environment.yaml --prefix ./envs/SDP_PO
activeer:
conda activate ./envs/SDP_PO

Notitie: Dit is inmiddels getest op één laptop. Er waren veel problemen met dependencies en het initieren van een jupyter kernel voor de nieuwe environment. Dit is opgelost met een aantal dependencies (nb_conda_kernels, ipykernel, beide zonder versie specificatie). Het kan zijn dat een ander systeem tegen vergelijkbare of nieuwe issues aanloopt.

3. Run de volgende command:
```
mkdir data
```
4. Download de [Google Play Store Apps dataset](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps) en de [Parquet file](https://drive.google.com/drive/folders/1Yus7axpUms3iB6brn6_JRfAFDnkLTGeG). Voeg deze toe aan de nieuwe data map in de repo.

5. Klaar voor gebruik!
