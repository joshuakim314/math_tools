# HEADER
# Remember to code Pearson correlation for population.

from matrix import *
from math import *


def pearson_correlation_sample(X, Y):
    if len(X) != len(Y):
        print("Size of datasets do not match.")
        return False
    sum_of_product = 0
    sum_of_X = 0
    sum_of_Y = 0
    sum_of_X_squared = 0
    sum_of_Y_squared = 0
    n = len(X)
    for i in range(n):
        sum_of_product += X[i]*Y[i]
        sum_of_X += X[i]
        sum_of_Y += Y[i]
        sum_of_X_squared += X[i]**2
        sum_of_Y_squared += Y[i]**2
    return (n*sum_of_product - sum_of_X*sum_of_Y) / (sqrt(n*sum_of_X_squared - sum_of_X**2) * sqrt(n*sum_of_Y_squared - sum_of_Y**2))


def multiple_correlation(*args, response=None, is_sample=True, use_vector=False):
    args = list(args)
    if response is None:
        response = args.pop()
        print("Warning: explicit response data not provided. The last inputted dataset is used as the response data.")
    correlation_matrix_data = []
    if use_vector:
        correlation_vector_data = [truncate_decimal(pearson_correlation_sample(args[i], response)) for i in range(len(args))]
        correlation_vector = Matrix(len(args), 1, correlation_vector_data)
        for i in range(len(args)):
            for j in range(len(args)):
                correlation_matrix_data.append(truncate_decimal(pearson_correlation_sample(args[i], args[j])))
                # if correlation_matrix_data[-1] > 1:
                #     correlation_matrix_data[-1] = 1  # change this
        correlation_matrix = Matrix(len(args), len(args), correlation_matrix_data)
        correlation_squared = correlation_vector.get_transpose() * correlation_matrix.get_inverse() * correlation_vector
        return correlation_squared.get_store_data()[0] ** 0.5
    args.insert(0, response)
    for i in range(len(args)):
        for j in range(len(args)):
            correlation_matrix_data.append(truncate_decimal(pearson_correlation_sample(args[i], args[j])))
            # if correlation_matrix_data[-1] > 1:
            #     correlation_matrix_data[-1] = 1  # change this
    correlation_matrix = Matrix(len(args), len(args), correlation_matrix_data)
    correlation_cofactor = correlation_matrix.get_cofactor(0, 0)
    correlation_coefficient_squared = 1 - (correlation_matrix.get_determinant() / correlation_cofactor)
    return correlation_coefficient_squared ** (0.5)


def linear_least_square_fit(*args, response=None):
    if response is None:
        args = list(args)
        response = args.pop()
        print("Warning: explicit response data not provided. The last inputted dataset is used as the response data.")
    response_matrix = Matrix(len(response), 1, response)
    data_matrix = Matrix(len(response), 1, [1 for _ in range(len(response))])
    for M in args:
        data_matrix = matrix_augment(data_matrix, Matrix(len(M), 1, M))
    data_matrix_transpose = data_matrix.get_transpose()
    coefficient_matrix = matrix_solve_equation(
        data_matrix_transpose * data_matrix, data_matrix_transpose * response_matrix, False)   # solve X for (AT)AX = (AT)b
    coefficient_matrix_data = []
    try:
        coefficient_matrix_data = coefficient_matrix.get_store_data()
    except:
        pass
    return coefficient_matrix_data    # the coefficient order is as follows: [1, x1, x2, ..., xn]


def truncate_decimal(value, truncation_place=10):
    value_temp = int((10**truncation_place) * value)
    return float(value_temp) / (10**truncation_place)


if __name__ == '__main__':
    Y_perpen = [0.180555556, 0.1875, 0.180555556, 0.166666667, 0.225490196, 0.254901961, 0.254901961, 0.264705882]
    X_perpen = [450.487272, 411.5456, 361.278732, 320.549708, 462.4322, 430.3542, 411.5456, 395.169312]
    print(linear_least_square_fit(X_perpen, Y_perpen))

    Y_parallel = [0.175438596, 0.122807018, 0.078947368]
    X_parallel = [447.3164, 368.3892, 390.8906]
    print(linear_least_square_fit(X_parallel, Y_parallel))














    # print(pearson_correlation_sample([100,290,345,234],[145,623,825,876]))
    # print(pearson_correlation_sample([212,100,423,765],[145,623,825,876]))
    # print(pearson_correlation_sample([556,555,931,345],[145,623,825,876]))
    # print(multiple_correlation([100,290,345,234],[100,290,0,234],response=[145,623,825,876], use_vector=True))
    # print(truncate_decimal(12.3456789, 3))
