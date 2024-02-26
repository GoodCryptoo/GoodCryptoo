import requests
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import pandas as pd
import time
import asyncio
from telegram import Bot

# Remplacez 'TOKEN' par le token de votre bot Telegram
TOKEN = '6614674853:AAGbGExt_u4czIjqaauCGNG73Cz-7x_hgpM'
bot = Bot(token=TOKEN)

async def send_message(chat_id, text):
    await bot.send_message(chat_id=chat_id, text=text)

# Utilisation de la fonction pour envoyer un message
chat_id = '6584346888'  # Vous pouvez obtenir votre ID de chat en discutant avec @userinfobot sur Telegram

class Portefeuille:
  montant = 0

  def __init__(self, solde_initial):
      self.solde = solde_initial
      self.quantite = 0

  async def acheter(self, montant, prix):
      quantite_achetee = (self.solde * 0.5 / prix)
      self.solde -= quantite_achetee * prix
      self.quantite += quantite_achetee
      message = f"Vous avez acheté {quantite_achetee} unités à {prix} $/unité pour un total de {self.quantite * prix} $. Votre solde est maintenant de {self.solde} $."
      await send_message(chat_id, message)

  async def vendre(self, montant, prix):
      quantite_vendue = (self.solde * 0.5 / prix)
      self.solde += self.quantite * prix
      message_v = f"Vous avez vendu {quantite_vendue} unités à {prix} $/unité pour un benefice de {self.quantite * prix} $. Votre solde est maintenant de {self.solde} $."
      await send_message(chat_id, message_v)
      self.quantite -= self.quantite

  async def afficher_solde(self):
      print("Solde du portefeuille:", self.solde)

# Fonction pour récupérer les données historiques d'une crypto-monnaie depuis CoinGecko
def get_historical_data(symbol, days):
  end_date = datetime.now()
  start_date = end_date - timedelta(days=days)
  end_date = int(end_date.timestamp())
  start_date = int(start_date.timestamp())
  url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range?vs_currency=usd&from={start_date}&to={end_date}"
  response = requests.get(url)
  data = response.json()
  prices = [entry[1] for entry in data['prices']]
  return prices

def get_crypto_price(symbol):
  url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
  response = requests.get(url)
  data = response.json()
  return data[symbol]['usd']

# Fonction pour prédire la variation en pourcentage sur une période de 24 heures
def predict_percentage_variation_24h(historical_data):
  X = np.arange(1, len(historical_data) + 1).reshape(-1, 1)
  y = historical_data

  model = LinearRegression()
  model.fit(X, y)

  # Prédiction pour le dernier point de données de la tranche de 24 heures
  next_price = model.predict(np.array([[len(historical_data)]]))

  return next_price, model

# Fonction pour diviser les données historiques en tranches de 24 heures
def split_data_into_24h_slices(historical_data):
  slices = []
  slice_length = 96  # Nombre d'heures dans une tranche
  num_slices = len(historical_data) // slice_length

  for i in range(num_slices):
      slice_start = i * slice_length
      slice_end = slice_start + slice_length
      slices.append(historical_data[slice_start:slice_end])

  return slices

# Fonction pour prédire les mouvements sur une période de 24 heures
async def predict_movement_24h(symbol, days, risk_level, portefeuille, purchased_price):
  current_price = get_crypto_price(symbol)
  historical_data = get_historical_data(symbol, days)
  if len(historical_data) < days:
      print("Erreur: Impossible de récupérer suffisamment de données historiques.")
      exit()

  # Diviser les données historiques en tranches de 24 heures
  slices = split_data_into_24h_slices(historical_data)

  for slice_data in slices:
      predicted_price, model = predict_percentage_variation_24h(slice_data)

      # Correction: Utiliser la quantité de la classe Portefeuille
      action, potential_profit = determine_action(current_price, purchased_price, predicted_price, risk_level, portefeuille.quantite)

      movement_direction, movement_end_time = predict_movement_end(slice_data)

      # Calculer le RMSE en ne considérant que les données historiques de la tranche
      rmse = np.sqrt(mean_squared_error(slice_data[:-1], model.predict(np.arange(1, len(slice_data)).reshape(-1, 1))))

      # Calculer le coefficient de détermination (R²) en ne considérant que les données historiques de la tranche
      r2 = r2_score(slice_data[:-1], model.predict(np.arange(1, len(slice_data)).reshape(-1, 1)))

      # Afficher les informations pour la tranche de 24 heures
      print("Valeur actuelle:", current_price)
      print("Valeur prédite:", predicted_price)
      print("Action recommandée:", action)
      print("Direction du mouvement:", movement_direction)
      print("Fin du mouvement:", movement_end_time)
      print("RMSE:", rmse)
      print("R²:", r2)
      print("Bought:", purchased_price)
      print("Quantité de crypto detenue:", portefeuille.quantite)

      # Simulation du suivi de l'indication
      if action == "BUY":
          await portefeuille.acheter(portefeuille.solde, purchased_price)
          purchased_price = current_price
          print("Achat effectué.")
      elif action == "SELL":
          await portefeuille.vendre(portefeuille.solde, current_price)
          print("Vente effectuée.")

      # Afficher le solde actuel du portefeuille
      await portefeuille.afficher_solde()
      print()

# Fonction pour déterminer l'action à prendre en fonction de la prédiction et de l'état actuel du portefeuille
def determine_action(current_price, purchased_price, predicted_price, risk_level, amount):
  potential_profit = current_price - purchased_price
  if amount <= 0:
      if current_price < predicted_price:
          action = "BUY"
          purchased_price = get_crypto_price(symbol)
          return action, 0
      else:
          action = "HOLD"
          return action, 0
  else:
      if purchased_price >= current_price:
          action = "HOLD"
          return action, 0
      else:
          action = "SELL"
          return action, potential_profit

# Fonction pour prédire la fin du mouvement de la crypto-monnaie
def predict_movement_end(prices):
  # Convertir les prix en un DataFrame pandas
  df = pd.DataFrame({'price': prices})
  df['index'] = np.arange(len(df))

  # Ajuster une régression linéaire avec une fenêtre mobile sur les prix historiques
  X = df['index']  # Utiliser uniquement l'index comme variable exogène
  y = df['price']
  lowess = sm.nonparametric.lowess(y, X, frac=0.1)  # Fraction de points à utiliser pour lissage
  lowess_x = list(zip(*lowess))[0]
  lowess_y = list(zip(*lowess))[1]

  # Trouver l'indice où la tendance change
  trend_change_index = np.argmax(np.diff(lowess_y) > 0)

  # Convertir le résultat en un type d'entier natif
  trend_change_index = int(trend_change_index)

  # Trouver l'heure de la fin du mouvement en supposant que chaque pas de temps est de 1 heure
  movement_end_time = datetime.now() + timedelta(hours=trend_change_index)

  # Déterminer la direction du mouvement en comparant le dernier prix observé au dernier prix lissé
  if prices[-1] > lowess_y[-1]:
      movement_direction = "hausse à long terme"
  else:
      movement_direction = "baisse à long terme"

  return movement_direction, movement_end_time

# Récupération des informations pour la crypto-monnaie "beam"
symbol = "dogecoin"
days = 1  # Période de données historiques (en jours)
risk_level = 1  # Niveau de risque

# Solde initial du portefeuille
solde_initial = 50  

# Initialisation du portefeuille
portefeuille = Portefeuille(solde_initial)

# Prix d'achat initial de la crypto-monnaie
purchased_price = get_crypto_price(symbol)

# Fonction principale pour prédire les mouvements et agir en conséquence
async def main():
  while True:
      await predict_movement_24h(symbol, days, risk_level, portefeuille, purchased_price)
      await asyncio.sleep(60)  # Pause d'une minute entre chaque itération

# Lancement de la fonction principale
asyncio.run(main())
