import numpy as np
from random import shuffle

class Artificial_Neural_Network:
    """
    Tato třída implementuje jednoduchou umělou neuronovou síť.
    """

    def __init__(self, layers):
        """
        Inicializace neuronové sítě, především jejích vah a prahů.
        """
        self.layers = layers # Seznam počtu neuronů v jednotlivých po sobě jdoucích vrstvách
        self.L = len(self.layers) # Počet vrstev
        self.weights = [None] # Seznam matic vah všech vrstevv
        self.biases = [None] # Seznam sloupcových vektorů prahů neuronů všech vrstev

        # Náhodná inicializace vah a prahů
        for l in range(1, self.L):
            self.biases.append(np.random.randn(self.layers[l], 1))
            self.weights.append(np.random.randn(self.layers[l], self.layers[l-1]))

    def forward_propagation(self, x):
        """
        Tato metoda implementuje dopřednou propagaci neuronové sítě.
        """
        self.neuron_activations = [x] # Seznam sloupcových vektorů aktivací neuronů všech vrstev
        self.weighted_sums = [None] # Seznam vážených součtů neuronů všech vrstev

        current_layer_A = x # Sloupcový vektor aktivací neuronů první vrstvy
        for l in range(1, self.L):

            current_layer_W = self.weights[l] # Matice vah pro spojení neuronů l-té vrstvy s neurony (l-1)-té vrstvy
            previous_layer_A = current_layer_A # Sloupcový vektor aktivací neuronů l-té vrstvy
            current_layer_B = self.biases[l] # Sloupcový vektor prahů neuronů l-té vrstvy

            # Vztah (8)
            current_layer_Z = np.dot(current_layer_W, previous_layer_A) + current_layer_B 
            current_layer_A = self.logistic_activation(current_layer_Z)

            # Přidání vážených součtů a aktivací do paměti
            self.weighted_sums.append(current_layer_Z)
            self.neuron_activations.append(current_layer_A)

    def stochastic_gradient_descent(self, X_train, y_train, epochs, mini_batch_size, learning_rate, validation_data = None):
        """
        Tato metoda implementuje stochastický gradientní sestup.
        """

        # Historie učení pro analýzu průběhu učení
        self.history = {"loss": [], "accuracy": [], "val_accuracy": [], "val_loss": []} if validation_data else {"loss": [], "accuracy": []}

        for e in range(epochs):
            training_data = list(zip(X_train, y_train))
            shuffle(training_data) # Promíchání dat pro každou epochu učení
            h = None

            # Rozdělení na množiny mini_batch
            mini_batches = []
            for mb_index in range(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[mb_index:mb_index+mini_batch_size])

            for mini_batch in mini_batches:
                for x, y in mini_batch:
                    self.forward_propagation(x) # Dopředná propagace
                    h = self.neuron_activations[self.L-1].T[0] # Výstup neuronové sítě
                    nabla_wb_x = self.backward_propagation(h, y) # Zpětná propagace nabla je znak pro gradient

                    nabla_w_x = nabla_wb_x[0] # Gradient chyby vzhledem k váhám
                    nabla_b_x = nabla_wb_x[1] # Gradient chyby vzhledem k prahům

                    # Aktualizace parametrů
                    for l in range(1, self.L):
                        self.weights[l] -= learning_rate * nabla_w_x[l]/len(mini_batch) # Vztah (16)
                        self.biases[l] -= learning_rate * nabla_b_x[l]/len(mini_batch) # Vztah (17)

            # Spočítání přesnosti a chybové funkce na trénovacích datech pro danou epochu učení
            hs = []
            for x in X_train:
                self.forward_propagation(x)
                h = self.neuron_activations[self.L-1].T[0]
                hs.append(h)
            predictions = np.array(hs).argmax(axis=1)
            target_values = y_train.argmax(axis=1)
            training_accuracy = sum([1 for i in range(len(hs)) if predictions[i] == target_values[i]])/len(hs)
            training_loss = self.loss_function(hs, y_train)

            # Přidání do historie
            self.history["loss"].append(training_loss)
            self.history["accuracy"].append(training_accuracy)

            # Spočítání přesnosti a chybové funkce na validačních datech pro danou epochu učení
            if validation_data:
                X_val = validation_data[0]
                y_val = validation_data[1]
                hs = []
                for x in X_val:
                    self.forward_propagation(x)
                    h = self.neuron_activations[self.L-1].T[0]
                    hs.append(h)
                predictions = np.array(hs).argmax(axis=1)
                target_values = y_val.argmax(axis=1)
                validation_accuracy = sum([1 for i in range(len(hs)) if predictions[i] == target_values[i]])/len(hs)
                validation_loss = self.loss_function(hs, y_val)

                # Přidání do historie
                self.history["val_loss"].append(validation_loss)
                self.history["val_accuracy"].append(validation_accuracy)

                # Vytisknutí výsledků epochy učení do terminálu v připadě vložených validačních dat
                print(f"Epoch {e+1} -> Training loss: {training_loss}, Training accuracy: {training_accuracy}, Validation loss: {validation_loss}, Validation accuracy: {validation_accuracy}")
            else:
                # Vytisknutí výsledků epochy učení do terminálu
                print(f"Epoch {e+1} -> Loss: {training_loss}, Accuracy: {training_accuracy}")
        
    def backward_propagation(self, h, y):
        """
        Tato metoda implementuje zpětnou propagaci neuronové sítě.
        """
        nabla_a = [] # Gradient chyby vzhledem k aktivacím všech neuronů
        last_layer_nabla_a = np.array([[self.loss_derivative(hi, yi)] for hi, yi in zip(h, y)]) # Gradient chyby vzhledem k aktivacím neuronů výstupní vrstvy

        nabla_a.append(last_layer_nabla_a)
        current_layer_nabla_a = last_layer_nabla_a
        for l in range(self.L-1, 0, -1):
            previous_layer_nabla_a = [] # Gradient chyby vzhledem k aktivacím neuronů (l-1)-té vrstvy
            current_layer_Z = self.weighted_sums[l] # Sloupcový vektor vážených součtů neuronů l-té vrstvy
            current_layer_W = self.weights[l] # Matice vah pro spojení neuronů l-té vrstvy s neurony (l-1)-té vrstvy
            previous_layer_nabla_a = np.dot(current_layer_W.T, self.logistic_derivative(current_layer_Z) * current_layer_nabla_a) # Vztah (29)
            nabla_a.append(previous_layer_nabla_a) # Přidání do gradientu chyby vzhledem ke všem aktivacím
            current_layer_nabla_a = previous_layer_nabla_a # Gradient chyby vzhledem k aktivacím neuronů l-té vrstvy
        nabla_a.reverse() # Přehození špatného pořadí

        nabla_b, nabla_w, current_layer_nabla_b, current_layer_nabla_w = [None], [None], [], []
        for l in range(1, self.L):
            current_layer_Z = self.weighted_sums[l] # Sloupcový vektor vážených součtů neuronů l-té vrstvy
            previous_layer_A = self.neuron_activations[l-1] # Sloupcový vektor aktivací neuronů (l-1)-té vrstvy
            current_layer_nabla_b = self.logistic_derivative(current_layer_Z)*nabla_a[l] # Vztah (28)
            current_layer_nabla_w = np.dot(current_layer_nabla_b, previous_layer_A.T) # Vztah (27)
            nabla_b.append(current_layer_nabla_b) # Přidání do gradientu chyby vzhledem ke všem prahům
            nabla_w.append(current_layer_nabla_w) # Přidání do gradientu chyby vzhledem ke všem váhám
        
        return (nabla_w, nabla_b)
    
    def predict(self, X):
        """
        Tato metoda implementuje možnost použití naučené neuronové sítě na nových datech.
        """
        hs = []
        for x in X:
            self.forward_propagation(x)
            h = self.neuron_activations[self.L-1].T[0]
            hs.append(h)
        return np.array(hs)
    
    def evaluate(self, X_test, y_test):
        """
        Tato metoda implementuje testování neuronové sítě.
        """
        hs = []
        for x in X_test:
            self.forward_propagation(x)
            h = self.neuron_activations[self.L-1].T[0]
            hs.append(h)
        predictions = np.array(hs).argmax(axis=1)
        target_values = y_test.argmax(axis=1)
        test_loss = self.loss_function(hs, y_test)
        test_accuracy = sum([1 for i in range(len(hs)) if predictions[i] == target_values[i]])/len(hs)
        print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

    def logistic_activation(self, Z):
        """Logistická aktivační funkce"""
        return 1/(1+np.exp(-Z)) # Vztah (4)

    def logistic_derivative(self, z):
        """Derivace logistické aktivační funkce"""
        return self.logistic_activation(z)*(1-self.logistic_activation(z)) # Vztah (5)
    
    def loss_function(self, H, Y):
        """Chybová funkce MSE"""
        return sum([np.dot(Y[i]-H[i], Y[i]-H[i]) for i in range(len(H))])/len(H) # Vztah (9)

    def loss_derivative(self, hi, yi):
        """Parciální derivace jedné chyby vzhledem k hi."""
        return 2*(hi-yi) # Součást vztahů (18), (19) a (20)