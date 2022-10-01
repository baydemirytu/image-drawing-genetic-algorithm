import imageio.v2 as iio
from pathlib import Path

import numpy
import numpy as np
import random
import matplotlib.pyplot as plt


def save_path_image(name, path_matrix):  # gelen matrisi png olarak keydeder
    iio.imwrite(str(name) + '.png', path_matrix)


def initialize_genes(array):  # ilk adımdaki gen dizisini rastgele atar
    # array matrisinin her bir satırı bir kromozomu, kromozomoun elemanı ise bir movements elemanını temsil eder
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j] = random.randint(0, 7)
        array[i] = check_correct_gene(array[i])  # sınırdan dışarı çıkma gibi durumları düzeltir

    return array


def create_path_matrix(gene_array):  # gen dizisinin ifade ettiği resim matrisini oluşturur
    location = [row_count - 1, 0]  # sol alttan başlamak için
    gene_array = check_correct_gene(gene_array)
    path_matrix = np.zeros((row_count, col_count), numpy.uint8)
    path_matrix[location[0]][location[1]] = 255

    for i in range(solution_length):
        location = np.add(location, movements[gene_array[i]])
        path_matrix[location[0]][location[1]] = 255

    return path_matrix


def look_around(location):  # etrafındaki geçerli noktaları bulur (sınırdan dışarı çıkmayan)
    valid_points = []
    for i in range(8):
        tmp_location = np.add(location, movements[i])
        if row_count > tmp_location[0] >= 0 and col_count > tmp_location[1] >= 0:
            valid_points.append(i)
    return valid_points


def check_correct_gene(gene_array):  # gendeki istenmeyen durumlrı düzelten ana method
    location = [row_count - 1, 0]
    for i in range(solution_length):
        tmp_move = gene_array[i]
        tmp_location = np.add(location, movements[gene_array[i]])
        valid = False
        iteration = 0
        while not valid and iteration < 20:

            if row_count > tmp_location[0] >= 0 and col_count > tmp_location[1] >= 0:
                valid = True
            else:
                pool = look_around(location)

                tmp_move = random.choice(pool)

                tmp_location = np.add(location, movements[tmp_move])

            iteration += 1
        if valid:
            location = tmp_location
            gene_array[i] = tmp_move
        else:
            gene_array[i] = 8

    return gene_array


def detect_turn_badness(gene_array):  # toplam dönüş açısının kötülüğünüü hesaplar.
    badness = 0
    for i in range(1, solution_length):
        badness += int(abs(gene_array[i-1] - gene_array[i])) % 5
    return int(badness*0.1)  # katsayısı değiştirilerek daha iyi sonuçlar alınabilir


def fitness_function1(gene_array):  # beyaz renkler aynı ise yakınlaşır mantığındaki tahmin fonksiyonu
    path_matrix = create_path_matrix(gene_array)  # dışarıya taşmamış bir şekilde pathi düzeltip döndürür
    increase = 250  # detect_turn_badness den gelecek değer ile uyumlu çalışması için değiştirilebilir
    fitness = 0
    for i in range(row_count):
        for j in range(col_count):
            if image_arr[i][j] == path_matrix[i][j] == 255:
                fitness += increase
    answer = fitness - detect_turn_badness(gene_array)
    if answer < 0:  # negatif iyilik değerini engeller
        return 0
    return answer


def fitness_function2(gene_array):  # renkleri aynı ise yakınlaşır mantığındaki tahmin fonksiyonu
    path_matrix = create_path_matrix(gene_array)  # dışarıya taşmamış bir şekilde pathi düzeltip döndürür
    increase_white = 1000
    increase_black = 1  # siyahların aynılık oranı beyaz ile eşit olmamamlı. beyaz daha değerli
    fitness = 0
    for i in range(row_count):
        for j in range(col_count):
            if image_arr[i][j] == path_matrix[i][j] == 255:
                fitness += increase_white
            elif image_arr[i][j] == path_matrix[i][j] == 0:
                fitness += increase_black
    answer = fitness - detect_turn_badness(gene_array)
    if answer < 0:  # negatif iyilik değerini engeller
        return 0
    return answer


def single_point_crossover(array1, array2):  # tek noktalı çaprazlama
    first_part = int(solution_length / 4)
    first_gene = np.zeros(solution_length, int)
    second_gene = np.zeros(solution_length, int)

    for i in range(first_part):
        first_gene[i] = array1[i]
    for i in range(first_part, solution_length):
        first_gene[i] = array2[i]
    for i in range(first_part):
        second_gene[i] = array2[i]
    for i in range(first_part, solution_length):
        second_gene[i] = array1[i]

    return first_gene, second_gene


def mutate(array, parent):  # gelen geni mutasyona uğratır
    if parent:  # gelen gen direk bir sonraki jenerasyona aktarılacaksa sadece bir hareketi değişir
        array[random.randint(0, solution_length) % len(array)] = random.randint(0, 7)
    else:
        mutation_num = int(solution_length * 0.05)
        if mutation_num == 0:
            mutation_num = 1

        for cell in range(mutation_num):
            array[random.randint(0, solution_length) % len(array)] = random.randint(0, 7)

    return array


def find_max_index(array):  # o ana kadar ki jenerasyonlar içinde en iyi bireyin indexini bulur
    max_index = -1
    max_value = -1
    for i in range(len(array)):
        if array[i] > max_value:
            max_index = i
            max_value = array[i]
    return max_index


imageList = []

movements = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [0, 0]]
# sol-üst, yukarı, sağ-üst,    sağ,   sağ-alt,  alt,   sol-alt, sol, sabit


for file in Path("images/").iterdir():
    # for döngüsü directoryde ne kadar resim varsa hepsini tek çalışma zamanında işlemesi için tasarlanmıştır.
    # daha sonra vakit yetersizliği nedeniyle kaldırılamamıştır. programın çalışmasına engel değildir
    file_name = file.name.split(".")[0]
    imageList.append(iio.imread(file))

    image = imageList[-1]
    image_arr = np.array(image, numpy.uint8)
    image_arr = np.array(image_arr[:, :, 0])  # deleting unnecessary dimensions

    row_count = image_arr.shape[0]
    col_count = image_arr.shape[1]

    solution_length = int(np.count_nonzero(image_arr == 255) * 3)  # her birey/gen/kromozom için çözüm uzunluğu
    generation_number = 480  # bir resim için üretilecek jenerasyon sayısı
    population = 60  # populasyondaki birey sayısı
    keep_parents = int(population / 3)  # bir sonraki jenereasyona crossover olmadan aktarılacak gen sayısı
    # keep_parents, population ın bazı değerleri için program index out of bound vermektedir :(
    crossover_number = population - keep_parents  # crossover a uğrayacak gen sayısı

    genes = np.zeros((population, solution_length), int)
    initialize_genes(genes)  # ilk adımdaki rastgele çözümlerin üretilmesi
    print(image_arr)

    fitness_by_generation = []  # her jenerasyondaki en iyi çözümün değerini tutar
    mean_of_fitness = np.zeros(generation_number)  # jenerasyonların ortalama iyilik değerlerini tutar

    for gen in range(generation_number):
        print(gen)
        genes_fitness = []  # bir popülasyon içindeki tüm genlerin iyilik değerini tutar : ör: [5,12500]
        for chromosome in range(population):
            # değerlendirme fonksiyonu buradan değiştirilebilir
            genes_fitness.append([chromosome, fitness_function2(genes[chromosome])])

        # popülasyondaki sırasını kaybetmeden fitness değerine göre
        genes_fitness.sort(key=lambda x: x[1], reverse=True)

        # bir sonraki jenersayonun boş olarak atanması
        new_population = np.zeros((population, solution_length), int)

        for i in range(keep_parents):  # en iyi üretimlerin sonraki nesle direkt aktarılması
            new_population[i] = genes[genes_fitness[i][0]]

        max = 0  # rulet tekeri için toplam fitness değerinin bulunması
        selection_probs = []  # her gen için olasılıkların tutulduğu dizi
        for i in range(len(genes_fitness)):
            max += genes_fitness[i][1]

        mean_of_fitness[gen] = max/population

        for i in range(len(genes_fitness)):
            selection_probs.append(genes_fitness[i][1] / max)

        # direkt aktarılmayacak olan genlerin rulet tekeri ile seçilerek single point crossover yapılması
        for j in range(keep_parents, population, 2):

            random_index_1 = np.random.choice(population, p=selection_probs)
            random_index_2 = np.random.choice(population, p=selection_probs)
            tmp1, tmp2 = single_point_crossover(genes[random_index_1], genes[random_index_2])
            new_population[j] = tmp1
            new_population[j + 1] = tmp2

        fitness_by_generation.append(genes_fitness[0][1])  # bu jenerasyonun en iyisinin eklenmesi

        print('max fitness of gen ' + str(gen) + ': ' + str(genes_fitness[0][1]))
        print('max fitness until now is ' + str(find_max_index(fitness_by_generation)) + '. gen and its fitness is: ' + str(np.max(fitness_by_generation)))
        print('------')

        # her jenerasyonun en iyi resminin klasöre kaydedilmesi
        save_path_image(file_name + '_' + str(gen), create_path_matrix(genes[genes_fitness[0][0]]))

        for i in range(keep_parents):  # parentlara tek noktadan mutasyon
            new_population[i] = mutate(new_population[i], parent=True)

        for i in range(keep_parents, population - 1):  # normallere mutasyon
            new_population[i] = mutate(new_population[i], parent=True)

        genes = new_population  # yeni jenerasyonun asıl olarak atanması

    x = np.arange(stop=generation_number, step=1)
    plt.plot(x, fitness_by_generation, label="best")
    plt.plot(x, mean_of_fitness, label="mean")
    plt.show()
