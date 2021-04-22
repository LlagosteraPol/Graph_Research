__author__ = 'Pol Llagostera Blasco'

from collections import Counter
from collections import OrderedDict
from math import factorial

import itertools as itt
import numpy as np
import sympy


class Utilities(object):

    @staticmethod
    def number_combination(n, k):
        """
        Determines how many combinations can be done.
        :param n: Number of elements.
        :param k: Number of elements to combine.
        :return: Number of combinations with the given specifications.
        """
        if k < 0:
            return 0

        return factorial(n) / factorial(k) / factorial(n - k)

    @staticmethod
    def remove_dup_with_order(text):
        """
        Remove the duplicate letters of the given string keeping the same order of characters.
        :param text: String to remove the duplicate characters.
        :return: String without duplicate characters.
        """
        return "".join(OrderedDict.fromkeys(text))

    @staticmethod
    def remove_dup_without_order(text):
        """
        Remove the duplicate characters of the given string without any order.
        :param text: String to remove the duplicate characters.
        :return: String without duplicate characters.
        """
        return "".join(set(text))

    @staticmethod
    def intersection(lst1, lst2):
        """
        Gives the intersection between the two given lists. Complexity O(n)
        :param lst1: First list
        :param lst2: Second list
        :return: A list containing the intersection of the given lists
        """
        # Use of hybrid method
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    @staticmethod
    def difference(lst1, lst2):
        """
        Gives the difference between the two given lists
        :param lst1: First list
        :param lst2: Second list
        :return: A list containing the diffenrence of the given lists
        """
        return list(set(lst1) - set(lst2))

    @staticmethod
    def set_combinations(sets, n, incl_excl_intersect=False):
        """
        Combination of the elements of the given sets.
        :param sets: Sets to combine.
        :param n: Num of elements to combine xCn
        :param incl_excl_intersect: Special flag for inclusion_exclusion algorithm
        :return: List of all combinations.
        """
        raw_combinations = []

        for subset in itt.combinations(sets, n):
            raw_combinations.append(subset)

        combinations = []

        for raw_comb in raw_combinations:
            if incl_excl_intersect:
                raw_comb = list(itt.chain.from_iterable(raw_comb))  # Join lists
            raw_comb = set().union(raw_comb)  # Remove duplicate elements
            combinations.append(raw_comb)

        return combinations

    @staticmethod
    def subtract_dict_values(dict1, dict2):
        """
        Subtraction of the two given dictionaries.
        :param dict1: First dictionary to subtract.
        :param dict2: Second dictionary to subtract.
        :return: Dictionary product of the subtraction of the given dictionaries.
        """
        for key in dict1:
            if key in dict2:
                dict1[key] = dict1[key] - dict2[key]
        return dict1

    @staticmethod
    def dict_union(dict1, dict2):
        """
        Union of the two given dictionaries.
        :param dict1: First dictionary to union.
        :param dict2: Second dictionary to union.
        :return: Dictionary product of the union of the given dictionaries.
        """
        xx = Counter(dict1)
        yy = Counter(dict2)
        xx.update(yy)

        return xx

    @staticmethod
    def flatten_nested_list(lst):
        """
        Flatten the given list.
        :param lst: List to be flattened.
        :return: Flattened list.
        """
        return [y for x in lst for y in x]

    @staticmethod
    def list_to_dict(lst, value):
        """
        Convert the given list to a dictionary where all the keys have the given value.
        :param lst: List to be converted to dictionary (the values of the list will
        be the keys).
        :param value: The value of all the keys.
        :return: Dictionary where keys-> lst and value-> value.
        """
        return {k: value for k in lst}

    @staticmethod
    def compare_coefficients(coefficients1, coefficients2):
        """
        Compare each coefficient of the first polynomial with each coefficient of the second one.
        If all the coefficients of the first polynomial are greater than the ones of the second polynomial
        then return True, else return False
        :param coefficients1: First polynomial coefficients
        :param coefficients2: Second polynomial coefficients
        :return: boolean
        """
        if len(coefficients1) != len(coefficients2):
            raise Exception("Both polynomials must have the same degree.")

        for i in range(0, len(coefficients1)):

            if abs(coefficients1[i]) < abs(coefficients2[i]):
                if coefficients1[i] == coefficients2[i] and coefficients1 != 0:
                    return False

        return True

    @staticmethod
    def polynomial2binomial(polynomial):
        """
        This method transforms the given polynomial to its binomial form.
        :param polynomial: Polynomial to convert to binomial form
        :return: Binomial Polynomial, coefficients
        """
        p = sympy.symbols('p')

        # Assuring that the given polynomial is in the right class
        if type(polynomial) is not sympy.polys.polytools.Poly:
            polynomial = sympy.poly(polynomial)

        binomial = 0
        degree = sympy.degree(polynomial, p)

        # Get coefficients (and round the ones really close to 0)
        coefficients = Utilities.refine_polynomial_coefficients(polynomial)
        coefficients = np.trim_zeros(coefficients)  # Delete all right zeroes
        n_coeff = len(coefficients)

        # Get binomial coefficients
        aux = n_coeff
        aux_degree = degree
        for i in range(1, n_coeff):
            coeff2expand = coefficients[-i]
            expanded = sympy.Poly(coeff2expand * p ** (aux_degree - n_coeff + 1) * (1 - p) ** (aux - 1))
            tmp_coefficients = expanded.all_coeffs()
            tmp_coefficients = np.trim_zeros(tmp_coefficients)
            tmp_n_coeff = len(tmp_coefficients)

            for z in range(2, tmp_n_coeff + 1):
                coefficients[(-z - (n_coeff - tmp_n_coeff))] -= tmp_coefficients[-z]

            aux -= 1
            aux_degree += 1

        # Assemble binomial polynomial
        aux_degree = degree
        for coeff in coefficients:
            binomial += coeff * p ** aux_degree * (1 - p) ** (degree - aux_degree)
            aux_degree -= 1

        return binomial, coefficients

    @staticmethod
    def refine_polynomial_coefficients(polynomial):
        """
        This method will round the coefficients of the polynomial that are almost zero (ex. at the order of e-10).
        When calculating Rel(G,p), if some of the coefficients are like this maybe is due to noise when calculating
        large amounts reliabilities with big polynomials.
        :param polynomial: polynomial to refine
        :return: refined polynomial
        """
        coefficients = polynomial.all_coeffs()

        refined_coefficients = list()
        for coefficient in coefficients:
            refined_coefficients.append(round(coefficient, 0))

        return refined_coefficients
