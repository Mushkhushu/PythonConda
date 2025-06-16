import matplotlib.pyplot as plt
import numpy as np
# refacto, passer toutes les variables nécessaires en paramètres, sauf la lib que l'on importe
def display_predict_diff_errors(y_test,predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(range(20), y_test[:20], label='Valeur réelle', marker='o')
    plt.plot(range(20), predictions[:20], label='Prédiction', marker='x')
    plt.title("Erreurs")
    plt.xlabel("Index dans l'ensemble de test")
    plt.ylabel("annulations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Sauvegarder la figure au format PNG
    plt.savefig("rfr_predict_error.png", dpi=300, bbox_inches='tight')
    plt.show()

def display_predict_diff_errors2(y_test, predictions, n=20, save_path=None):
    """
    Visualise les écarts entre les valeurs réelles et les prédictions.

    Paramètres :
    - y_test : valeurs réelles (array-like ou Series)
    - predictions : valeurs prédites (array-like)
    - n : nombre de premières valeurs à afficher (défaut : 20)
    - save_path : chemin pour sauvegarder l'image (ex: "plot.png") ou None pour ne pas sauvegarder
    """

    # Conversion en array pour indexation sûre
    y_test = np.array(y_test)
    predictions = np.array(predictions)

    # Vérification taille
    if len(y_test) != len(predictions):
        raise ValueError("y_test et predictions doivent avoir la même longueur.")

    # Limiter à n premiers éléments
    y_test_subset = y_test[:n]
    predictions_subset = predictions[:n]

    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(range(n), y_test_subset, label='Valeur réelle', marker='o', color='green')
    plt.plot(range(n), predictions_subset, label='Prédiction', marker='x', color='orange')
    plt.title(f"Comparaison Réel vs Prédiction (sur {n} échantillons)")
    plt.xlabel("Index dans l'ensemble de test")
    plt.ylabel("Annulation (0 ou 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📷 Graphique sauvegardé dans : {save_path}")

    plt.show()