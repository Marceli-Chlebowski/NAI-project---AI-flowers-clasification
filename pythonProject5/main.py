import math

# Funkcja do obliczania odległości euklidesowej między dwoma punktami
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Funkcja do wczytywania danych treningowych z pliku
def read_training_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Usuń białe znaki i podziel wiersz za pomocą tabulatora
            attributes_with_label = line.strip().split('\t')
            # Pobierz pierwsze cztery wartości jako atrybuty
            attributes = [float(attr.replace(',', '.')) for attr in attributes_with_label[:4]]  # Konwersja na liczby zmiennoprzecinkowe
            label = attributes_with_label[4].strip()  # Usunięcie białych znaków
            data.append((attributes, label))
    return data

# Funkcja do wczytywania danych testowych z pliku
def read_test_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Usuń białe znaki i podziel wiersz za pomocą tabulatora
            attributes_with_label = line.strip().split('\t')
            # Pobierz pierwsze cztery wartości jako atrybuty
            attributes = [float(attr.replace(',', '.')) for attr in attributes_with_label[:4]]  # Konwersja na liczby zmiennoprzecinkowe
            label = attributes_with_label[4].strip()  # Usunięcie białych znaków
            data.append((attributes, label))
    return data

# Funkcja klasyfikująca przykłady z użyciem k-NN
def classify_knn(training_data, test_data, k):
    correct_count = 0
    for test_point, test_label in test_data:
        # Inicjalizacja listy k najbliższych sąsiadów
        nearest_neighbors = [(float('inf'), '')] * k  # Początkowo ustawiamy odległości na nieskończoność

        # Obliczenie odległości między testowym punktem a każdym punktem treningowym
        for train_point, train_label in training_data:
            distance = euclidean_distance(test_point, train_point)

            # Aktualizacja listy k najbliższych sąsiadów, jeśli nowa odległość jest mniejsza od maksymalnej w liście
            max_distance = float('-inf')
            max_distance_index = -1
            for i, (d, _) in enumerate(nearest_neighbors):
                if distance < d:
                    if d > max_distance:
                        max_distance = d
                        max_distance_index = i
            if max_distance_index != -1:
                nearest_neighbors[max_distance_index] = (distance, train_label)

        # Zliczenie etykiet wśród k najbliższych sąsiadów
        label_count = {}
        for distance, label in nearest_neighbors:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # Wybór najczęściej występującej etykiety
        max_label = ''
        max_count = 0
        for lbl, count in label_count.items():
            if count > max_count:
                max_label = lbl
                max_count = count

        # Sprawdzenie, czy przewidziana etykieta zgadza się z rzeczywistą etykietą
        if max_label == test_label:
            correct_count += 1

    return correct_count


# Funkcja do obliczania dokładności klasyfikacji w procentach
def calculate_accuracy(correct_count, total_count):
    return (correct_count / total_count) * 100


if __name__ == "__main__":
    # Wczytanie danych treningowych i testowych
    training_data = read_training_data("iris_training.txt")
    test_data = read_test_data("iris_test.txt")

    # Wczytanie parametru k od użytkownika
    k = int(input("Podaj wartość parametru k: "))

    # Klasyfikacja danych testowych
    correct_count = classify_knn(training_data, test_data, k)
    total_count = len(test_data)

    # Wyświetlenie wyników
    print("Liczba prawidłowo zaklasyfikowanych przykładów:", correct_count)
    print("Dokładność klasyfikacji:", calculate_accuracy(correct_count, total_count), "%")

    # Klasyfikacja nowego punktu wprowadzonego przez użytkownika
    while True:
        print("\nWprowadź dane dla nowego kwiatka (w formacie: długość działki kielicha, szerokość działki kielicha, długość płatka, szerokość płatka cyferki oddzielj kropka np 1.5 1.4 4.4 6.6):")
        new_point = [float(x) for x in input().split()]
        if len(new_point) != 4:
            print("Nieprawidłowa liczba atrybutów. Spróbuj ponownie.")
            continue

        # Klasyfikacja nowego punktu
        nearest_neighbors = [(float('inf'), '')] * k  # Początkowo ustawiamy odległości na nieskończoność
        for train_point, train_label in training_data:
            distance = euclidean_distance(new_point, train_point)

            # Aktualizacja listy k najbliższych sąsiadów, jeśli nowa odległość jest mniejsza od maksymalnej w liście
            max_distance = float('-inf')
            max_distance_index = -1
            for i, (d, _) in enumerate(nearest_neighbors):
                if distance < d:
                    if d > max_distance:
                        max_distance = d
                        max_distance_index = i
            if max_distance_index != -1:
                nearest_neighbors[max_distance_index] = (distance, train_label)

        # Zliczenie etykiet wśród k najbliższych sąsiadów
        label_count = {}
        for distance, label in nearest_neighbors:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # Wybór najczęściej występującej etykiety
        max_label = ''
        max_count = 0
        for lbl, count in label_count.items():
            if count > max_count:
                max_label = lbl
                max_count = count

        print("Gatunek przewidywany dla wprowadzonego kwiatka:", max_label)
