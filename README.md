# Voorspellen data Google Play Store Apps dataset :construction_worker:

## Nieuws
- [24/02/2023]: Feature engineering klaar!!
- [20/03/2023]: Repo gemaakt, yay

##TODO:
- Verify environment instantiation. Currently the anaconda way has worked once on the requirements.txt file, but only when commenting out the fastparquet package. Trying to install with fastparquet took long (~10 min) and resulted in dependency conflicts.

## Introductie
Data Science opdracht voor Smart Data People. Hier wordt gebruikt gemaakt van de [Google Play Store Apps dataset](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps). 

## Resultaten

:trollface:

## Opstarten

**Let op:** Nog niet af, dus kan nog veel veranderen

1. Kloon de repo :shipit:

2. Installeer de dependencies:
2.1 Optioneel, maar best practice:
maak een (conda) environment aan speciaal voor dit project.
In VS Code open je de cloned repo, dit is je workspace. Dan ctrl+shift+p -> 'Python: Create Environment' -> Keuze voor Conda wordt verder aangenomen, maar Venv kan ook -> Als het goed is staat er nu in je repo een .conda folder, dit kun je in VS Code selecteren als kernel/interpreter (.ipynb files en .py files resp.). Het staat dan in de lijst als '.\.conda\python.exe'.

Als VS Code bij het runnen van een .ipynb notebook aangeeft dat de kernel niet geïnstalleerd is en de tooltip werkt niet, gebruik dan deze command in de terminal:
conda install -c anaconda ipykernel                

Environment maken en selecteren kan ook met terminal commands gedaan worden. Werkt niet (altijd) met powershell, vermoedelijk wanneer anaconda niet aan path is toegevoegd t.t.v. installatie.
2.2
```
Heb je een conda environment gemaakt:
conda install --file requirements.txt -c conda-forge
Anders:
pip install -r requirements.txt
```
3. Run de volgende command:
```
mkdir data
```
4. Download de [Google Play Store Apps dataset](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps) en de [Parquet file](https://drive.google.com/drive/folders/1Yus7axpUms3iB6brn6_JRfAFDnkLTGeG). Voeg deze toe aan de nieuwe data map in de repo.

5. ??? :rage3:
