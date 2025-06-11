import matplotlib.pyplot as plt
# refacto, passer toutes les variables nécessaires en paramètres, sauf la lib que l'on importe
def display_predict_diff_errors(y_test,predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(range(20), y_test[:20], label='Valeur réelle', marker='o')
    plt.plot(range(20), predictions[:20], label='Prédiction', marker='x')
    plt.title("Erreurs")
    plt.xlabel("Index dans l'ensemble de test")
    plt.ylabel("Ticket Vendu")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Sauvegarder la figure au format PNG
    plt.savefig("rfr_predict_error.png", dpi=300, bbox_inches='tight')
    plt.show()