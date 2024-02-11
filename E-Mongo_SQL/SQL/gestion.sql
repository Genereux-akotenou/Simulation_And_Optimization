-- Active: 1704878946883@@127.0.0.1@3306@ATELIER_SQL
-- CREATE TABLE commande_ligne(
--     id int NOT NULL AUTO_INCREMENT, 
--     commande_id int NOT NULL,
--     nom varchar(255) NOT NULL,
--     quantite int NOT NULL,
--     prix_unitaire DOUBLE NOT NULL,
--     prix_total DOUBLE NOT NULL,
--     PRIMARY KEY (id),
--     FOREIGN KEY (commande_id) REFERENCES commande(id)
-- )

-- INSERT INTO client VALUES <<copy and pass the dataset>>

# Obtenir le client ayant le prénom “Muriel” et le mot de passe “test11”, sachant que l’encodage du mot de passe est effectué avec l’algorithme Sha1
SELECT * FROM `client` WHERE prenom = 'Muriel' AND password = SHA1('test11');

# Obtenir et trier par ordre décroissant sur les noms la liste de tous les clients dont l'id est strictement supérieur à 10.
SELECT * FROM `client` WHERE id > 10 ORDER BY nom DESC;

# Obtenir la liste des clients (nom et prénom) et leurs commandes (reférence et date). Afficher la liste précédente uniquement pour les commandes passées à partir du '2019-02-08'.
SELECT nom, prenom, reference, date_achat
FROM `client`
INNER JOIN commande ON client.id = commande.client_id
WHERE commande.date_achat >= '2019-02-08';

SELECT nom, prenom, reference, date_achat
FROM `client`, commande
WHERE commande.date_achat >= '2019-02-08' AND client.id = commande.client_id;

# On va chercher à obtenir des résultats issus de ces 3 tables, une "simple" jointure ne sera donc pas suffisante, il va falloir créer 2 jointures, une pour relier les tables une pour relier les tables commande et commande_ligne et une autre pour relier les tables commande et client.
SELECT client.nom, client.prenom, date_achat, commande_id, quantite, prix_unitaire FROM `client`
INNER JOIN commande ON client.id = commande.id
INNER JOIN commande_ligne ON commande.id = commande_ligne.commande_id;

#- Obtenir la liste des achats effectués par Maris Buisson. On veut afficher la date des achats, le nom des produits, la quantité et le prix unitaire de chaque produit.
SELECT date_achat, commande_ligne.nom, quantite, prix_unitaire 
FROM `client` 
INNER JOIN commande ON client.id = commande.id 
INNER JOIN commande_ligne ON commande.id = commande_ligne.commande_id WHERE client.nom = "Buisson" AND prenom = "Maris";

# En utilisant la fonction SUM, déterminer le nombre total d'articles commandés par M. Saunier.
SELECT SUM(quantite) AS "TOTAL COUNT ARTICLES"
FROM client
INNER JOIN commande ON client.id = commande.client_id
INNER JOIN commande_ligne ON commande.id = commande_ligne.commande_id
WHERE client.nom = "Saunier";

# 7 - En utilisant la commande SQL UPDATE, mettre à jour le prix total à l’intérieur de chaque ligne des commandes, en fonction du prix unitaire et de la quantité.
UPDATE commande_ligne
SET prix_total = prix_unitaire*quantite;

# 8 - déterminer le montant total des achats de M. Jung.
SELECT SUM(prix_total) AS "TOTAL PRICE ARTICLES"
FROM client
INNER JOIN commande ON client.id = commande.client_id
INNER JOIN commande_ligne ON commande.id = commande_ligne.commande_id
WHERE client.nom = "Jung";

# 9 - Obtenir la liste de tous les produits qui sont présent sur plusieurs commandes. (Pour chaque nom de produit, combien de fois est présent dans les commandes)
SELECT nom, COUNT(nom)
FROM commande_ligne
GROUP BY nom
HAVING COUNT(nom) > 1;

# 10 - Obtenir la liste de tous les produits qui sont présent sur plusieurs commandes et y ajouter une colonne qui liste les identifiants des commandes associées.
é