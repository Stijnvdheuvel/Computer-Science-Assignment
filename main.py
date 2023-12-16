import re
import json
import math
import numpy as np
import editdistance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import heapq
import itertools
import time
from functools import lru_cache
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import os

np.random.seed(12345)


# ------------------------- CLEANING DATA ---------------------------------


def replace_with_rounded(match):
    """ Rounds up values in inch units found in product titles. """
    value_str = match.group(1)
    # If the value doesn't have a decimal point, insert one after the first two digits
    if '.' not in value_str:
        value_str = value_str[:2] + '.' + value_str[2:]
    value = float(value_str)
    rounded_value = math.ceil(value)
    return f"{rounded_value}inch"


def clean_and_normalize(text, extensions, is_title):
    """ Normalizes text by standardizing units and cleaning it for further processing. """
    # Step 1: Transform units into standardized format
    unit_transformations = {
        r'\b(\d+(\.\d+)?)\s*(?:inch|inches|"|-inch| inch|  inch|in)\b': r'\1inch ',
        r'\b(\d+(\.\d+)?)\s*(?:hertz|hertz|Hz|HZ| hz|-hz|hz)\b': r'\1hz ',
        r'\b(\d+(\.\d+)?)\s*(?:lb| lb|-lb|Lb|pounds)\b': r'\1lbs ',
        r'\b(\d+(\.\d+)?)\s*(?:mm| mm|-mm)\b': r'\1mm ',
        r'\b(\d+(\.\d+)?)\s*(?:watts|watt|Watt)\b': r'\1watt ',
        r'\b(\d+(\.\d+)?)\s*(?:volt)\b': r'\1v ',
        r'\b(\d+(\.\d+)?)\s*(?:cd m2|cdm2|cd m| cdm)\b': r'\1cdm '
    }

    for pattern, replacement in unit_transformations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Replace common words and characters with spaces # FROM PAPER FLAVIUS - EXTENSION
    common_replacements = {
        r'\b(?:and|or)\b': ' ',  # Replaces 'and' or 'or' only when they are standalone words
        r'(?<=\s)[&/\\-](?=\s)': ' ',  # Replace '&', '/', '\', '-' only if surrounded by spaces
        r'\s+x\s+': ' ',  # Replace 'x' only if surrounded by spaces
        r'\b(?:class)\b': ' '  # Replaces 'and' or 'or' only when they are standalone words
    }
    for pattern, replacement in common_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE) if extensions else text

    # Additional replacements --> can this be moved into top part?
    text = re.sub(r'\"', 'inch', text)
    text = re.sub(r'\u00a0', ' ', text)
    text = re.sub(r'\u0099', ' ', text)
    text = re.sub(r'&#176', 'deg', text)
    text = re.sub(r'\u00b0', 'deg', text)
    text = re.sub(r'\u00ba', 'deg', text)
    text = re.sub(r'\u00b2', ' ', text)
    text = re.sub(r'\u00c2', ' ', text)
    text = re.sub(r'\bcd m\b', 'cdm', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcd m2\b', 'cdm', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpounds\b', 'lbs', text, flags=re.IGNORECASE)
    text = re.sub(r'\blb\b', 'lbs', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwatts\b', 'watt', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d(?:\s?\d)?\s?)w\b', r'\1watt', text,
                  flags=re.IGNORECASE) if extensions else text  # replace w in digit-w and digit-space-w by watts

    # Step 2: Replace upper-case characters with lower-case characters
    text = text.lower()

    # Step 3: Remove spaces and non-alphanumeric tokens in front of units
    text = re.sub(r'[^\w\d]*(inch|hz|lbs|watt|kg|mm|ms|v|cdm)\b', r'\1', text, flags=re.IGNORECASE)

    # Extension:
    if extensions and is_title:
        inch_pattern = re.compile(r'\b(\d+\.\d+|\d{2,}\.\d|\d{4,})inch\b')
        text = inch_pattern.sub(replace_with_rounded, text)
        text = re.sub(r'\s+', ' ', text)

    return text


def get_all_brands(data, brand_file_path_input):
    """ Extracts and saves all unique brand names from the data. """
    # Read existing brands from the file
    with open(brand_file_path_input, 'r') as txt_file:
        existing_brands = {line.strip().lower() for line in txt_file}

    # Update the set with brand information from the dataset
    all_brands_set = existing_brands.copy()

    for product_data in data.values():
        for product_entry in product_data:
            keys = product_entry.get('featuresMap', {}).keys()
            for key in keys:
                if re.search(r'brand', key, re.IGNORECASE):
                    all_brands_set.add(product_entry.get('featuresMap').get(key, '').lower())

    all_brands_set = sorted(all_brands_set)
    # Some double: LG, LG ELECTRONICS JVC, JVC TV
    # One : PANASONIC, PANSONIC # CHOSEN TO IGNORE
    # Appear double but not actually double : TPV TECHNOLOGY, TP VISION, PYE, PYLE, COMPAL, COMPAQ

    # Save the updated set to a file
    with open(brand_file_path_input, "w") as txt_file:
        txt_file.write("\n".join(all_brands_set))

    # print("all brands: ", all_brands_set)
    return all_brands_set


def get_brand(product_entry, all_brands):
    """ Determines the brand of a product from its 'brand' feature, 'title, or 'url'. """
    keys = product_entry.get('featuresMap', {}).keys()

    # Search for a key containing the text 'brand' in featuresMap
    for key in keys:
        if re.search(r'brand', key, re.IGNORECASE):
            return product_entry.get('featuresMap').get(key, '')

    # If nothing is found, check whether the title contains a string from all_brands
    title = product_entry.get('title', '').lower()
    for brand in all_brands:
        if brand.lower() in title:
            return brand

    # If still nothing is found, check the URL
    url = product_entry.get('url', '').lower()
    for brand in all_brands:
        if brand.lower() in url:
            return brand

    # If nothing is found, print a message
    print("No brand information found for product:", product_entry)
    return None


def read_data(json_file_path_):
    """ Reads JSON data from a file. """
    with open(json_file_path_, 'r') as file:
        return json.load(file)


def save_to_json(data, file_path):
    """ Saves data to a JSON file. """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def clean_and_normalize_data(data, brand_file_path_input, extensions):
    """ Applies cleaning and normalization to the entire dataset. """
    all_brands = get_all_brands(data, brand_file_path_input)
    target_index_mapping_ = {}
    target_index_counter = 0

    for product_data in data.values():
        for product_entry in product_data:
            title = product_entry.get('title', '')
            cleaned_title = clean_and_normalize(title, extensions, is_title=True)
            product_entry['title'] = cleaned_title

            features_map = product_entry.get('featuresMap', {})
            for key, value in features_map.items():
                cleaned_value = clean_and_normalize(value, extensions, is_title=False)
                features_map[key] = cleaned_value

            product_entry['brand_name_new'] = get_brand(product_entry, all_brands)
            product_entry['target_index'] = target_index_counter
            target_index_mapping_[target_index_counter] = [product_entry]
            target_index_counter += 1

    return data, target_index_mapping_, target_index_counter


def clean_and_save_data(original_file_path_input, brand_file_path_input, clean_file_path_input,
                        clean_file_path_extension_input):
    """ Cleans original data with and without extensions and saves it as a new JSON file. """
    # Read the original data
    data_original = read_data(original_file_path_input)

    # Clean and normalize data without extensions
    cleaned_data, target_index_mapping, final_index = \
        clean_and_normalize_data(data_original, brand_file_path_input, extensions=False)
    save_to_json(target_index_mapping, clean_file_path_input)
    print(f"Cleaned and normalized dataset saved to {clean_file_path_input}")

    # Clean and normalize data with extensions
    cleaned_data_extension, target_index_mapping_extension, final_index_extension = \
        clean_and_normalize_data(data_original, brand_file_path_input, extensions=True)
    save_to_json(target_index_mapping_extension, clean_file_path_extension_input)
    print(f"Cleaned and normalized dataset with extensions saved to {clean_file_path_extension_input}")


# -------------------------------- BOOTSTRAPPING ALGORITHM ----------------------------------

def find_band_row_combination(k_minhash_input, t_value_):
    """ Finds the closest possible combination of bands and rows for LSH given a t-value. """
    closest_solution = None
    min_diff = float('inf')

    for bands_temp in range(1, k_minhash_input + 1):
        if k_minhash_input % bands_temp == 0:
            rows_per_band_temp = k_minhash_input // bands_temp
            t_approx_temp = (1 / bands_temp) ** (1 / rows_per_band_temp)
            diff = abs(t_approx_temp - t_value_)

            if diff < min_diff:
                min_diff = diff
                closest_solution = (bands_temp, rows_per_band_temp, t_approx_temp)

    return closest_solution


def extract_model_words(parameter, pattern_compiled):
    """ Extracts model words from a string based on a compiled regex pattern. """
    matches = pattern_compiled.findall(parameter)
    return {match[0] for match in matches}


def round_up_numerical_part(word):
    """ Rounds up the numerical part of a KVP model word. """
    numerical_part_match = pattern_numerical_part_compiled.search(word)
    if numerical_part_match:
        numerical_part = numerical_part_match.group()
        rounded_number = math.ceil(float(numerical_part))
        return word.replace(numerical_part, str(rounded_number))
    return word


def get_values(product, pattern_compiled, round_numerical_kvp):
    """ Extracts model words from product features for Key Value Pairs. """
    model_words_kvp_func = set()
    features_map_values_func = product.get('featuresMap', {}).values()
    for value_func in features_map_values_func:
        model_words_in_features_func = extract_model_words(value_func, pattern_compiled)
        for word_func in model_words_in_features_func:
            numerical_part_func = pattern_numerical_part_compiled.search(word_func)
            if numerical_part_func:
                if round_numerical_kvp:  # EXTENSION
                    model_words_kvp_func.add(math.ceil(float(numerical_part_func.group())))
                else:
                    model_words_kvp_func.add(numerical_part_func.group())
    return model_words_kvp_func


def get_values2(product):
    """ Extracts value strings from product features for Key Value Pairs. """
    features_map_values_func = product.get('featuresMap', {}).values()
    return features_map_values_func


def extract_unique_model_words(data, round_numerical_kvp):
    """ Extracts and returns unique model words from the dataset. """
    unique_mw_title_set = set()
    unique_mw_kvp_set = set()

    for product_data in data.values():
        for product_entry in product_data:
            title = product_entry.get('title', '')
            mw_in_title = extract_model_words(title, pattern_title_compiled)
            unique_mw_title_set.update(mw_in_title)

            features_map_values = product_entry.get('featuresMap', {}).values()
            for value in features_map_values:
                mw_in_features = extract_model_words(value, pattern_kvp_compiled)

                for word in mw_in_features:
                    numerical_part = pattern_numerical_part_compiled.match(word)
                    if numerical_part:
                        if round_numerical_kvp:  # EXTENSION
                            unique_mw_kvp_set.add(math.ceil(float(numerical_part.group())))
                        else:
                            unique_mw_kvp_set.add(numerical_part.group())

    return unique_mw_title_set, unique_mw_kvp_set


def process_data_mw(data, round_numerical_kvp):
    """ Processes the data to extract model words and determine a close k_minhash of ~50% of unique mw. """
    unique_mw_title, unique_mw_kvp = extract_unique_model_words(data, round_numerical_kvp)
    unique_mw = list(unique_mw_title)
    unique_mw.extend(list(unique_mw_kvp))
    num_unique_mw_title = len(unique_mw_title)
    k_minhash = int(round((0.5 * len(unique_mw)) / 20) * 20)
    print(len(unique_mw_kvp), len(unique_mw_title))

    return unique_mw, num_unique_mw_title, k_minhash


def create_binary_matrix(data, unique_mw_list, num_unique_mw_title, n_products, round_numerical_kvp):
    """ Creates a binary matrix representation of the data using Algorithm 1 of Hartveld et al. """
    num_unique_mw = len(unique_mw_list)
    binary_matrix_temp = np.zeros((num_unique_mw, n_products), dtype=int)
    j = 0

    for product_data in data.values():
        for product_entry in product_data:
            title = product_entry.get('title', '')
            mw_in_title = extract_model_words(title, pattern_title_compiled)
            mw_value_in_features = get_values(product_entry, pattern_kvp_compiled, round_numerical_kvp)
            mw_title_in_features = get_values(product_entry, pattern_title_compiled, round_numerical_kvp)

            binary_vector = np.zeros(num_unique_mw, dtype=int)
            for i, model_word in enumerate(unique_mw_list):
                if i < num_unique_mw_title and model_word in mw_in_title:
                    binary_vector[i] = 1
                elif i < num_unique_mw_title and model_word in mw_title_in_features:
                    binary_vector[i] = 1
                elif i >= num_unique_mw_title and model_word in mw_value_in_features:
                    binary_vector[i] = 1

            binary_matrix_temp[:, j] = binary_vector
            j += 1
    return binary_matrix_temp


def is_prime(num):
    """Check if a number is prime."""
    if num < 2:
        return False
    for i_prime in range(2, int(num ** 0.5) + 1):
        if num % i_prime == 0:
            return False
    return True


def generate_hash_functions(k):
    """ Generates k hash functions for creating minhash signatures. """
    hash_functions_list = []
    p_ = np.random.randint(2 * k, 3 * k)
    a_ = 0
    b_ = 0
    # Find a random prime number larger than k
    while not is_prime(p_):
        p_ += 1
    for _ in range(k):
        a_ = np.random.randint(1, k)
        b_ = np.random.randint(1, k)
        hash_functions_list.append(lambda x, a=a_, b=b_, p=p_: (a * x + b) % p)
    return hash_functions_list


def compute_signature_matrix(binary_matrix_input, k_minhash_):
    """ Computes the signature matrix from a binary matrix for LSH using minhashing. """
    hash_functions = generate_hash_functions(k_minhash_)
    num_columns = binary_matrix_input.shape[1]
    sig_matrix = np.full((k_minhash_, num_columns), np.inf)

    for row_index, binary_row in enumerate(binary_matrix_input):
        hash_values = np.array([h(row_index) for h in hash_functions])

        for col_index in range(num_columns):
            if binary_matrix_input[row_index, col_index] == 1:
                sig_matrix[:, col_index] = np.minimum(sig_matrix[:, col_index], hash_values)

    return sig_matrix


def locality_sensitive_hashing(signature_matrix_input, bands_input, rows_per_band_input):
    """ Applies Locality Sensitive Hashing to a signature matrix with specified bands and rows. """
    assert signature_matrix_input.shape[0] == bands_input * rows_per_band_input, \
        "Invalid choice of bands and rows_per_band "
    buckets = {}

    for band in range(bands_input):
        start_row = band * rows_per_band_input
        end_row = (band + 1) * rows_per_band_input

        for col_index in range(signature_matrix_input.shape[1]):
            hash_value = hash(tuple(signature_matrix_input[start_row:end_row, col_index]))
            if hash_value not in buckets:
                buckets[hash_value] = []
            buckets[hash_value].append(col_index)

    candidate_pairs_temp = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            candidate_pairs_temp.update((i, j) for i in bucket for j in bucket if i < j)
    return candidate_pairs_temp


def same_brand(product1, product2):
    """ Checks if two products are from the same brand. """
    brand1 = product1.get('brand_name_new', '')
    brand2 = product2.get('brand_name_new', '')
    # Return true if different brand
    return brand1.lower() == brand2.lower() or brand1.lower() in brand2.lower() or brand2.lower() in brand1.lower()


def same_shop(product1, product2):
    """ Checks if two products are from the same shop. """
    shop1 = product1.get('shop', '')
    shop2 = product2.get('shop', '')
    return shop1.lower() == shop2.lower()


@lru_cache(maxsize=None)
def calc_qgram_sim(str1, str2, q=3):
    """ Calculates the q-gram similarity between two strings with dummy variables at start and end of each string. """
    str1 = '#' * (q - 1) + str1 + '#' * (q - 1)  # Dummy variables at start and end of string
    str2 = '#' * (q - 1) + str2 + '#' * (q - 1)  # Dummy variables at start and end of string

    set1 = set([str1[i:i + q] for i in range(len(str1) - q + 1)])
    set2 = set([str2[i:i + q] for i in range(len(str2) - q + 1)])
    n1 = len(set1)
    n2 = len(set2)
    return (n1 + n2 - len(set1 ^ set2)) / (n1 + n2) if (n1 + n2) != 0 else 0


def get_mw_kvp_subset(product, nmk, pattern_compiled):
    """ Gets a subset of product KVP model words from a set of Key Value Pairs. """
    mw_set = set()
    for key in nmk:
        value = product.get('featuresMap').get(key, '')
        mw_set.update(pattern_compiled.findall(value))
    return mw_set


def calc_cosine_sim(title1, title2):
    """ Calculates the cosine similarity between two product titles. """
    words1 = set(re.findall(r'\b\w+\b', title1.lower()))
    words2 = set(re.findall(r'\b\w+\b', title2.lower()))
    intersection = len(words1.intersection(words2))
    denominator = np.sqrt(len(words1) * len(words2))
    cosine_sim = intersection / denominator if denominator != 0 else 0
    return cosine_sim


def calc_lv_sim(string1, string2):
    """ Calculates the normalized Levenshtein similarity between two strings. """
    abs_lv = editdistance.eval(string1, string2)
    lv_sim_strings = abs_lv / max(len(string1), len(string2))
    return lv_sim_strings


def calc_avg_lv_sim(set1, set2):
    """ Calculates the average Levenshtein similarity between sets of strings. """
    sum_lv_sim_sets = 0
    tot_sum = 0
    for x in set1:
        for y in set2:
            sum_x_y = len(x) + len(y)
            tot_sum += sum_x_y
            sum_lv_sim_sets += (1 - calc_lv_sim(x, y)) * sum_x_y

    avg_lv_sim_sets = sum_lv_sim_sets / tot_sum if tot_sum != 0 else 0
    return avg_lv_sim_sets


def calc_avg_lv_sim_mw(set1, set2, threshold):
    """ Calculates the average Levenshtein similarity between sets of model words over only model words that have
    approximately the same the non-numeric part and the numeric part is the same. """
    sum_lv_sim_sets = 0
    tot_sum = 0

    for x in set1:
        for y in set2:
            nonnumeric_x = re.sub(r'\d', '', x)
            nonnumeric_y = re.sub(r'\d', '', y)
            if calc_lv_sim(nonnumeric_x, nonnumeric_y) < threshold:
                numeric_x = re.sub(r'[^0-9]', '', x)
                numeric_y = re.sub(r'[^0-9]', '', y)
                sum_x_y = len(x) + len(y)
                tot_sum += sum_x_y
                sum_lv_sim_sets += (1 - calc_lv_sim(x, y)) * sum_x_y
                if numeric_x != numeric_y:
                    return None, -1

    if tot_sum > 0:
        numeric_check = 1
    else:
        numeric_check = 0

    avg_lv_sim_sets = sum_lv_sim_sets / tot_sum if tot_sum != 0 else 0

    return avg_lv_sim_sets, numeric_check


def get_title_sim(product1, product2, alpha, beta, threshold=0.5, delta=0.4):
    """ Calculates title similarity using the Title Model Words Method. """
    title1 = product1.get('title', '')
    title2 = product2.get('title', '')
    cos_similarity = calc_cosine_sim(title1, title2)

    if cos_similarity > alpha:
        return 1
    else:
        mw_title1 = extract_model_words(title1, pattern_title_compiled)
        mw_title2 = extract_model_words(title2, pattern_title_compiled)
        avg_lv_sim_mw, numeric_check = calc_avg_lv_sim_mw(mw_title1, mw_title2, threshold)

        if numeric_check == -1:  # approximately the same AND numeric not the same:
            return -1

        avg_lv_sim = calc_avg_lv_sim(mw_title1, mw_title2)  # 2. Calculate average Levenshtein ABOVE 1, NOT POSSIBLE
        title_sim = beta * cos_similarity + (1 - beta) * avg_lv_sim

        if numeric_check == 1:  # approximately the same AND numeric THE SAME
            title_sim = delta * avg_lv_sim_mw + (1 - delta) * title_sim
        return title_sim


def calc_dissimilarity(i, j, product1, product2, alpha, beta, gamma, mu, true_pairs_input):
    """ Calculates the final dissimilarity score between two products. """
    is_true_pair = (i, j) in true_pairs_input

    if same_shop(product1, product2) or not same_brand(product1, product2):
        dist_pair = np.inf
    else:
        sim = 0
        avg_sim = 0
        m = 0
        w = 0
        nmk1 = list(product1.get('featuresMap', {}).keys())
        nmk2 = list(product2.get('featuresMap', {}).keys())
        nmk1_copy = nmk1.copy()
        nmk2_copy = nmk2.copy()
        min_num_features = min(len(nmk1), len(nmk2))
        for key1 in nmk1_copy:
            if key1 not in nmk1:
                continue
            for key2 in nmk2_copy:
                if key2 not in nmk2 or key1 not in nmk1:
                    continue
                key_sim = calc_qgram_sim(key1, key2)
                value1 = product1.get('featuresMap').get(key1, '')
                value2 = product2.get('featuresMap').get(key2, '')
                if key_sim > gamma:
                    value_sim = calc_qgram_sim(value1, value2)
                    weight = key_sim
                    sim = sim + weight * value_sim
                    m = m + 1
                    w = w + weight
                    nmk1.remove(key1)
                    nmk2.remove(key2)

        if w > 0:
            avg_sim = sim / w
        mw_nmk1 = get_mw_kvp_subset(product1, nmk1,
                                    pattern_title_compiled)
        mw_nmk2 = get_mw_kvp_subset(product2, nmk2,
                                    pattern_title_compiled)

        mw_percentage = len(mw_nmk1.intersection(mw_nmk2)) / len(mw_nmk1.union(mw_nmk2)) \
            if len(mw_nmk1.union(mw_nmk2)) != 0 else 0
        title_sim = get_title_sim(product1, product2, alpha, beta)
        if title_sim == -1:
            theta1 = m / min_num_features
            theta2 = 1 - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_percentage
        else:
            theta1 = (1 - mu) * m / min_num_features
            theta2 = 1 - mu - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_percentage + mu * title_sim
        dist_pair = 1 - h_sim
        if dist_pair > 0.5 and is_true_pair and printer:
            print("------------ rejected true pair ------------")
            print("titles :", product1.get('title', ''), "|", product2.get('title', ''))
            print("values1, values2 :", get_values2(product1), get_values2(product2))
            print("dist_pair, pair :", dist_pair, i, j)
            print("avg_sim :", avg_sim)
            print("matching word percentage :", mw_percentage)
            print("hsim :", h_sim)
            print("title_sim :", title_sim)
            print("theta1, theta2, m, min_features, w :", theta1, theta2, m, min_num_features, w)
    return dist_pair


def get_dissimilarity_matrix(data, candidate_pairs_input, n_products_, alpha, beta, gamma, mu, true_pairs_input):
    """ Generates a full dissimilarity matrix from product data using given candidate pairs. """
    dissimilarity_scores_matrix = np.full((n_products_, n_products_), np.inf)
    np.fill_diagonal(dissimilarity_scores_matrix, 0)
    non_infinity_count = 0

    for i, j in candidate_pairs_input:
        product1, product2 = data[str(i)][0], data[str(j)][0]
        dissimilarity_score = calc_dissimilarity(i, j, product1, product2, alpha, beta, gamma, mu, true_pairs_input)
        if dissimilarity_score != np.inf:
            non_infinity_count += 1
        dissimilarity_scores_matrix[i, j] = dissimilarity_scores_matrix[j, i] = dissimilarity_score
    return dissimilarity_scores_matrix


def save_all_dissimilarity_matrices(data, data_name, parameter_ranges_input, true_pairs_input, n_products):
    """ Saves all dissimilarity matrices for different combinations of algorithm parameters, to run more
    efficient model optimization and evaluation at a data set of this size. For larger datasets, use
    get_dissimilarity_matrix only using the data and candidate pairs. """
    all_possible_pairs = get_all_pairs(data)
    base_dir = f'Matrices/{data_name}/'
    os.makedirs(base_dir, exist_ok=True)  # Create directory if it doesn't exist
    print("Starting first dissimilarity matrix")
    i = 0

    for alpha in parameter_ranges_input['alpha']:
        for beta in parameter_ranges_input['beta']:
            for gamma in parameter_ranges_input['gamma']:
                for mu in parameter_ranges_input['mu']:
                    key = (alpha, beta, gamma, mu)
                    print("Starting with key: ", key)
                    dissimilarity_matrix = get_dissimilarity_matrix(data, all_possible_pairs, n_products, alpha, beta,
                                                                    gamma, mu, true_pairs_input)
                    file_path = os.path.join(base_dir, f'{key}.npy')
                    np.save(file_path, dissimilarity_matrix)  # Save the matrix in .npy format
                    i = i + 1
                    print("Another dissimilarity matrix saved: ", i)


def load_dissimilarity_matrix(data_name, key):
    """ Loads a saved dissimilarity matrix based on the given key specified as (alpha, beta, gamma, mu) and
    data name. """
    base_dir = f'Matrices/{data_name}/'
    file_path = os.path.join(base_dir, f'{key}.npy')

    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"Matrix file {file_path} not found.")
        return None


def extract_sub_matrices(matrix, indices):
    """ Extracts a submatrix from the given full dissimilarity matrix based on specified indices. """
    submatrix = matrix[np.ix_(indices, indices)]
    return submatrix


def filter_dissimilarity_matrix(dissimilarity_matrix, candidate_pairs):
    """ Filters the dissimilarity matrix to only include dissimilarities for candidate pairs, setting all other
    values to infinity for [i,j] where i!=j, and [i,j]=0 where i=j."""
    # Initialize a new matrix with infinity values and zeros on the diagonal
    new_matrix = np.full(dissimilarity_matrix.shape, np.inf)
    np.fill_diagonal(new_matrix, 0)

    # Extract row and column indices from candidate pairs
    row_indices, col_indices = zip(*candidate_pairs)

    # Update the new matrix with values from the original dissimilarity matrix
    new_matrix[row_indices, col_indices] = dissimilarity_matrix[row_indices, col_indices]

    # Since the matrix is symmetric, update the transposed indices as well
    new_matrix[col_indices, row_indices] = dissimilarity_matrix[row_indices, col_indices]

    return new_matrix


def efficient_hierarchical_clustering(distance_matrix, epsilon):
    """
    Performs adapted single linkage hierarchical clustering on a distance matrix using a specified threshold
    (epsilon) with priority heaping. In final model it was chosen not to use this, and to instead use complete linkage.
    """
    n = len(distance_matrix)
    assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix must be symmetric"

    clusters = [{i} for i in range(n)]
    cluster_labels = np.arange(n)
    pq = []
    predicted_pairs_temp = []

    for i in range(n):
        for j in range(i + 1, n):
            heapq.heappush(pq, (distance_matrix[i, j], (i, j)))

    while len(clusters) > 1:
        if not pq:
            break

        min_distance, (min_i, min_j) = heapq.heappop(pq)
        if cluster_labels[min_i] != min_i or cluster_labels[min_j] != min_j:
            continue

        if min_distance > epsilon:
            break

        for index in clusters[min_j]:
            cluster_labels[index] = min_i
            predicted_pairs_temp.append((min_i, index))

        clusters[min_i] |= clusters[min_j]
        clusters[min_j] = set()

        for k in range(n):
            if k != min_i and cluster_labels[k] == k:
                new_dist = min(distance_matrix[min_i, k], distance_matrix[min_j, k])
                if np.isinf(new_dist):
                    new_dist = np.inf
                heapq.heappush(pq, (new_dist, (min_i, k)))

    # Filter out pairs within the same cluster
    predicted_pairs_temp = [pair for pair in predicted_pairs_temp if cluster_labels[pair[0]] == cluster_labels[pair[1]]]
    return predicted_pairs_temp


def agglomerative_hierarchical_clustering(distance_matrix, epsilon, linkage_type='complete'):
    """ Applies complete linkage agglomerative hierarchical clustering to a distance matrix/dissimilarity matrix with a
    specified threshold (epsilon). """
    n = len(distance_matrix)

    # Using AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                         linkage=linkage_type, distance_threshold=epsilon,
                                         compute_full_tree=True)

    # The fit_predict method returns the labels of the clusters
    distance_matrix[np.isinf(distance_matrix)] = 1e10
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Construct predicted pairs based on cluster labels
    predicted_pairs_temp = []
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_labels[i] == cluster_labels[j]:
                predicted_pairs_temp.append((i, j))

    return predicted_pairs_temp


def get_true_pairs(data):
    """ Identifies true duplicate pairs in the dataset based on matching model IDs. """
    model_pairs = {}
    for index, products in data.items():
        model_id = products[0]['modelID']
        if model_id not in model_pairs:
            model_pairs[model_id] = []
        model_pairs[model_id].append(int(index))
    return [tuple(pair) for pair_list in model_pairs.values() for pair in itertools.combinations(pair_list, 2)]


def get_all_pairs(data):
    """ Generates all possible pairs of data indices, for use in the save_all_dissimilarity_matrices calculations for
    efficient model optimization and evaluation for this size data set. Not recommended for larger data sets. """
    indices = sorted([int(index) for index in data.keys()])  # Extracting all the indices from the data
    all_pairs = list(combinations(indices, 2))  # Generating all possible pairs ordered i<j
    return all_pairs


def calculate_metrics(predicted_pairs_input, true_pairs, candidate_pairs):
    """ Calculates various metrics like Pair Quality, Pair Completeness, F1*-Score and F1-score for the algorithm. """
    # Star metrics - LSH
    dup_f = len(set(candidate_pairs) & set(true_pairs))  # duplicates found by LSH in candidate pairs
    dup_n = len(true_pairs)  # total number of duplicates in data
    num_c = len(candidate_pairs)  # number of comparisons (number of candidate pairs)
    pq_calc = dup_f / num_c  # Pair Quality
    pc_calc = dup_f / dup_n  # Pair Completeness
    sum_pq_pc = pq_calc + pc_calc
    f1_star_calc = 2 * pq_calc * pc_calc / sum_pq_pc if sum_pq_pc > 0 else 0  # F1-Star

    tp = 0
    fp = 0
    fn = 0
    for pair in predicted_pairs_input:
        if pair in true_pairs:
            tp += 1
        else:
            fp += 1
    for pair in true_pairs:
        if pair not in predicted_pairs_input:
            fn += 1

    precision_calc = tp / (tp + fp) if tp + fp > 0 else 0
    recall_calc = tp / (tp + fn) if tp + fn > 0 else 0
    sum_precision_recall = precision_calc + recall_calc
    f1_score_calc = 2 * precision_calc * recall_calc / sum_precision_recall if sum_precision_recall > 0 else 0  # F1

    metrics = {
        'pq': pq_calc,
        'pc': pc_calc,
        'f1_star': f1_star_calc,
        'precision': precision_calc,
        'recall': recall_calc,
        'f1': f1_score_calc
    }
    return metrics


def run_algorithm_steps(data, binary_matrix, bands, rows_per_band, k_minhash, alpha, beta, gamma, mu,
                        epsilon, true_pairs_input, cluster_extension, n_products, dissimilarity_matrix):
    """ Executes all the MSMP+, MSMP+ clean or MSMPE algorithm steps and calculates performance metrics and FOC. """
    tot_possible_comparisons = n_products * (n_products - 1) / 2
    true_pairs = get_true_pairs(data)

    signature_matrix = compute_signature_matrix(binary_matrix, k_minhash)
    candidate_pairs = locality_sensitive_hashing(signature_matrix, bands, rows_per_band)
    dissimilarity_matrix = filter_dissimilarity_matrix(dissimilarity_matrix, candidate_pairs)
    # dissimilarity_matrix = get_dissimilarity_matrix(data, candidate_pairs, n_products, alpha, beta, gamma, mu,
    #                                                 true_pairs)
    num_candidate_pairs = len(candidate_pairs)
    fraction_of_comparisons = num_candidate_pairs / tot_possible_comparisons

    if not cluster_extension:
        predicted_pairs = efficient_hierarchical_clustering(dissimilarity_matrix, epsilon)
    else:
        predicted_pairs = agglomerative_hierarchical_clustering(dissimilarity_matrix, epsilon)
    metrics = calculate_metrics(predicted_pairs, true_pairs, candidate_pairs)
    return metrics, fraction_of_comparisons


def optimize_parameters(data, binary_matrix, bands, rows_per_band, k_minhash, parameter_ranges_input,
                        true_pairs_input, cluster_extension, n_products, data_name, product_indices):
    """ Optimizes the algorithm parameters for the best performance on the given dataset. """
    optimal_params = {
        'alpha': parameter_ranges['alpha'][0],
        'beta': parameter_ranges['beta'][0],
        'gamma': parameter_ranges['gamma'][0],
        'mu': parameter_ranges['mu'][0],
        'epsilon': parameter_ranges['epsilon'][0],
    }
    optimal_metrics = {
        'pq': 0,
        'pc': 0,
        'f1_star': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    fraction = 0
    i = 0

    for alpha in parameter_ranges_input['alpha']:
        for beta in parameter_ranges_input['beta']:
            for gamma in parameter_ranges_input['gamma']:
                for mu in parameter_ranges_input['mu']:
                    key = (alpha, beta, gamma, mu)
                    full_dissimilarity_matrix = load_dissimilarity_matrix(data_name, key)
                    full_dissimilarity_matrix = extract_sub_matrices(full_dissimilarity_matrix, product_indices)
                    for epsilon in parameter_ranges_input['epsilon']:
                        i = i + 1
                        metrics, fraction = run_algorithm_steps(data, binary_matrix, bands,
                                                                rows_per_band, k_minhash, alpha, beta, gamma, mu,
                                                                epsilon, true_pairs_input, cluster_extension,
                                                                n_products, full_dissimilarity_matrix)
                        if metrics['f1'] > optimal_metrics['f1']:
                            optimal_params['alpha'] = alpha
                            optimal_params['beta'] = beta
                            optimal_params['gamma'] = gamma
                            optimal_params['mu'] = mu
                            optimal_params['epsilon'] = epsilon
                            optimal_metrics = metrics

    print("Optimal params: ", optimal_params)
    print("Optimal metrics: ", optimal_metrics)

    return optimal_params, optimal_metrics, fraction


def split_data(data, test_size=0.37):
    """ Splits the data into training and testing sets. """
    keys = list(data.keys())
    data_training_indices, data_test_indices = train_test_split(keys, test_size=test_size)
    train_data = {str(i): data[key] for i, key in enumerate(data_training_indices)}
    test_data = {str(i): data[key] for i, key in enumerate(data_test_indices)}
    return train_data, test_data, [int(item) for item in data_training_indices], [int(item) for item in
                                                                                  data_test_indices]


def run_bootstraps(data, data_name, n_bootstraps_input, t_range_input, parameter_ranges_input, cluster_extension=True,
                   round_numerical_kvp=False):
    """ Runs bootstrap experiments to evaluate the algorithm's performance over multiple iterations. """
    dict_bootstraps = {}

    for i_bootstrap in range(n_bootstraps_input):
        print("----------------- STARTING BOOTSTRAP ", i_bootstrap, "------------------")
        train_data, test_data, train_indices, test_indices = split_data(data)

        # extract unique model words and k_minhash for training data
        unique_mw_training, num_unique_mw_title_training, k_minhash_training = process_data_mw(train_data,
                                                                                               round_numerical_kvp)
        true_pairs_training = get_true_pairs(train_data)
        k_minhash_training = 1500 if not round_numerical_kvp else 900

        # extract unique model words and k_minhash for test data
        unique_mw_test, num_unique_mw_title_test, k_minhash_test = process_data_mw(test_data, round_numerical_kvp)
        true_pairs_test = get_true_pairs(test_data)
        k_minhash_test = 1100 if not round_numerical_kvp else 600

        # count number of products in each set
        n_products_training = sum(len(value) for value in train_data.values())
        n_products_test = sum(len(value) for value in test_data.values())

        # Get binary matrices now to improve runtime
        start_time = time.time()
        binary_matrix_train = create_binary_matrix(train_data, unique_mw_training, num_unique_mw_title_training,
                                                   n_products_training, round_numerical_kvp)
        binary_matrix_test = create_binary_matrix(test_data, unique_mw_test, num_unique_mw_title_test, n_products_test,
                                                  round_numerical_kvp)
        bin_time = time.time()
        print("Binary matrices found in: ", bin_time - start_time)

        # extract unique model words and k_minhash for test data
        for t_value in reversed(t_range_input):
            start_time = time.time()

            bands_training, rows_training, t_approx_training = find_band_row_combination(k_minhash_training, t_value)
            bands_test, rows_test, t_approx_test = find_band_row_combination(k_minhash_test, t_value)
            if np.abs(t_approx_training - t_value) >= 0.05:  # Necessary since otherwise overlap.
                print(f"t_approx_training not found for t_value: {t_value}")
                continue
            if np.abs(t_approx_test - t_value) >= 0.05:  # Necessary since otherwise overlap.
                print(f"t_approx_test not found for t_value: {t_value}")
                continue
            print("------------------------------------------------")
            print("We've started training for bands, rows, t_value, k_minhash :", bands_training, rows_training,
                  t_value, k_minhash_training)
            optimal_params, optimal_metrics, fraction_training = \
                optimize_parameters(train_data, binary_matrix_train, bands_training, rows_training, k_minhash_training,
                                    parameter_ranges_input, true_pairs_training, cluster_extension,
                                    n_products_training, data_name, train_indices)
            alpha, beta, gamma, mu, epsilon = optimal_params.values()
            # print("Optimal epsilon", epsilon, "for t", t_value)
            print("------------------------------------------------")
            print("We've started evaluating for bands, rows, t_value, k_minhash :", bands_test, rows_test, t_value,
                  k_minhash_test)
            time_opt = time.time() - start_time

            test_dissimilarity_matrix = load_dissimilarity_matrix(data_name, (alpha, beta, gamma, mu))
            test_dissimilarity_matrix = extract_sub_matrices(test_dissimilarity_matrix, test_indices)
            eval_metrics, fraction_test = \
                run_algorithm_steps(test_data, binary_matrix_test, bands_test, rows_test, k_minhash_test, alpha, beta,
                                    gamma, mu, epsilon, true_pairs_test, cluster_extension, n_products_test,
                                    test_dissimilarity_matrix)

            end_time = time.time()
            time_t = end_time - start_time

            # Store results in dictionary
            if t_value not in dict_bootstraps:
                dict_bootstraps[t_value] = []
            dict_bootstraps[t_value].append({
                "optimal_params": optimal_params,
                "optimal_metrics": optimal_metrics,
                "fraction_training": fraction_training,
                "t_approx_training": t_approx_training,
                "bands_training": bands_training,
                "rows_training": rows_training,
                "num_unique_mw_training": len(unique_mw_training),
                "eval_metrics": eval_metrics,
                "fraction_test": fraction_test,
                "t_approx_test": t_approx_test,
                "bands_test": bands_test,
                "rows_test": rows_test,
                "num_unique_mw_test": len(unique_mw_test),
                "time_opt": time_opt,
                "time": time_t,
            })
    return dict_bootstraps


# ----------------------- PLOTTING -----------------------------

def average_metrics_over_bootstraps(bootstrap_results, metric_key):
    """ Averages the specified metric over multiple bootstrap iterations. """
    averages = {}
    for t_value, results in bootstrap_results.items():
        metric_sum = sum(entry['eval_metrics'][metric_key] for entry in results)
        fraction_sum = sum(entry['fraction_test'] for entry in results)
        average_metric = metric_sum / len(results)
        average_fraction = fraction_sum / len(results)
        averages[t_value] = (average_fraction, average_metric)
    return averages


def plot_metric_vs_fraction_averages(metric_key, bootstrap_results1, bootstrap_results2=None,
                                     bootstrap_results3=None):
    """
    Plots the average of a specified metric against the fraction of comparisons for different algorithm extensions.
    """
    averages1 = average_metrics_over_bootstraps(bootstrap_results1, metric_key)
    averages2 = average_metrics_over_bootstraps(bootstrap_results2,
                                                metric_key) if bootstrap_results2 is not None else None
    averages3 = average_metrics_over_bootstraps(bootstrap_results3,
                                                metric_key) if bootstrap_results3 is not None else None

    # Extract fractions and metric values
    fractions1, y_values1 = zip(*sorted(averages1.values()))
    fractions2, y_values2, fractions3, y_values3 = 0, 0, 0, 0
    if bootstrap_results2 is not None:
        fractions2, y_values2 = zip(*sorted(averages2.values()))
    if bootstrap_results3 is not None:
        fractions3, y_values3 = zip(*sorted(averages3.values()))

    plt.figure(figsize=(10, 6))
    plt.plot(fractions1, y_values1, marker='o', linestyle='-', color='b', label='MSMP+')
    if bootstrap_results2 is not None:
        plt.plot(fractions2, y_values2, marker='s', linestyle='-', color='r', label='MSMP+ clean')
    if bootstrap_results3 is not None:
        plt.plot(fractions3, y_values3, marker='v', linestyle='-', color='g', label='MSMPE')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel(metric_key.capitalize())
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(plot_directory, f'{metric_key}_averages_plot.png')
    plt.savefig(plot_filename)
    plt.show()


def all_metrics_over_bootstraps(bootstrap_results, metric_key):
    """ Collects all metric values over multiple bootstrap iterations. """
    all_points = []
    for results in bootstrap_results.values():
        for entry in results:
            metric_ = entry['eval_metrics'][metric_key]
            fraction = entry['fraction_test']
            all_points.append((fraction, metric_))
    return sorted(all_points, key=lambda x: x[0])


def plot_metric_vs_fraction_all(metric_key, bootstrap_results1, bootstrap_results2=None,
                                bootstrap_results3=None, ):
    """ Plots all metric values against the fraction of comparisons for different algorithm variants. """
    all_points1 = all_metrics_over_bootstraps(bootstrap_results1, metric_key)
    all_points2 = all_metrics_over_bootstraps(bootstrap_results2,
                                              metric_key) if bootstrap_results2 is not None else None
    all_points3 = all_metrics_over_bootstraps(bootstrap_results3,
                                              metric_key) if bootstrap_results3 is not None else None

    fractions1 = [point[0] for point in all_points1]
    y_values1 = [point[1] for point in all_points1]
    fractions2, y_values2, fractions3, y_values3 = 0, 0, 0, 0
    if bootstrap_results2 is not None:
        fractions2 = [point[0] for point in all_points2]
        y_values2 = [point[1] for point in all_points2]
    if bootstrap_results3 is not None:
        fractions3 = [point[0] for point in all_points3]
        y_values3 = [point[1] for point in all_points3]

    plt.figure(figsize=(10, 6))
    plt.plot(fractions1, y_values1, marker='o', linestyle='-', color='b', label='MSMP+')
    if bootstrap_results2 is not None:
        plt.plot(fractions2, y_values2, marker='s', linestyle='-', color='r', label='MSMP+ clean')
    if bootstrap_results3 is not None:
        plt.plot(fractions3, y_values3, marker='v', linestyle='-', color='g', label='MSMPE')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel(metric_key)
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(plot_directory, f'{metric_key}_all_plot.png')
    plt.savefig(plot_filename)
    plt.show()


# ----------- Control parameters ----------------
t_range = np.arange(0.05, 1.0, 0.05)
n_bootstraps = 5

parameter_ranges = {
    'alpha': [0.602],  # not changed, sd close to 0
    'beta': [0.0],  # not changed, sd close to 0
    'gamma': [0.65, 0.75, 0.85],  # [0.65, 0.75, 0.85] hiermee beginnen
    'mu': [0.45, 0.55, 0.65, 0.75, 0.85],  # hiermee beginnen
    'epsilon': np.arange(0.4, 0.85, 0.05),  # 0.85 VAN MAKEN!!!! hiermee beginnen
}

clean_data_now = False  # True if you want to output the cleaned and normalized datasets.
printer = False  # True to turn debugging mode on
run_bootstraps_now = False  # True to run the bootstrapping algorithm
plot_now = True  # True to create the plots of performance measures against FOC
calc_matrices = False  # True if you want to recalculate dissimilarity_matrices

# ---------------------- Cleaning data ------------------------

original_file_path = 'TVs-all-merged.json'
brand_file_path = 'TV_brands.txt'
clean_file_path = 'cleaned_normalized_dataset.json'
clean_file_path_extension = 'cleaned_normalized_extension.json'
json_file_results = 'bootstrap-regular-results.json'
json_file_results2 = 'bootstrap-clean-results.json'
json_file_results3 = 'bootstrap-cluster-results.json'
plot_directory = "newest_plots"
result_directory = "results"

data_name_clean = 'clean'
data_name_extension = 'clean with extensions'

pattern_title_compiled = re.compile(r'\b([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)\b')
pattern_kvp_compiled = re.compile(r'\b(\d+(\.\d+)?[a-zA-Z]*)\b')
pattern_numerical_part_compiled = re.compile(r'\d+(\.\d+)?')

# ----------------------------- Main code
data_clean = read_data(clean_file_path)
data_clean_extension = read_data(clean_file_path_extension)

if clean_data_now:
    clean_and_save_data(original_file_path, brand_file_path, clean_file_path, clean_file_path_extension)

if calc_matrices:
    true_pairs_total = get_true_pairs(data_clean)
    n_products_total = sum(len(value) for value in data_clean.values())
    print("Starting calculation of full dissimilarity matrix regular")
    save_all_dissimilarity_matrices(data_clean, data_name_clean, parameter_ranges, true_pairs_total, n_products_total)
    print("Starting calculation of full dissimilarity matrix with extensions")
    save_all_dissimilarity_matrices(data_clean_extension, data_name_extension, parameter_ranges, true_pairs_total,
                                    n_products_total)

if run_bootstraps_now:
    result_directory = "results"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    print("Starting without extensions")
    dict_bootstraps_results = run_bootstraps(data_clean, data_name_clean, n_bootstraps, t_range, parameter_ranges,
                                             cluster_extension=True)
    result_file_path = os.path.join(result_directory, json_file_results)
    save_to_json(dict_bootstraps_results, result_file_path)

    print("Starting with data cleaning extensions")
    dict_bootstraps_results2 = run_bootstraps(data_clean_extension, data_name_extension, n_bootstraps, t_range,
                                              parameter_ranges,
                                              cluster_extension=True)
    result_file_path2 = os.path.join(result_directory, json_file_results2)
    save_to_json(dict_bootstraps_results2, result_file_path2)

    print("Starting with rounding up KVP extensions")
    dict_bootstraps_results3 = run_bootstraps(data_clean_extension, data_name_extension, n_bootstraps, t_range,
                                              parameter_ranges,
                                              cluster_extension=True, round_numerical_kvp=True)
    result_file_path3 = os.path.join(result_directory, json_file_results3)
    save_to_json(dict_bootstraps_results3, result_file_path3)

if plot_now:
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    result_file_path = os.path.join(result_directory, json_file_results)
    dict_bootstraps_results = read_data(result_file_path)
    result_file_path2 = os.path.join(result_directory, json_file_results2)
    dict_bootstraps_results2 = read_data(result_file_path2)
    result_file_path3 = os.path.join(result_directory, json_file_results3)
    dict_bootstraps_results3 = read_data(result_file_path3)

    metrics_to_plot = ['pq', 'pc', 'f1_star', 'f1']
    for metric in metrics_to_plot:
        plot_metric_vs_fraction_averages(metric, dict_bootstraps_results, dict_bootstraps_results2,
                                         dict_bootstraps_results3)
        plot_metric_vs_fraction_all(metric, dict_bootstraps_results, dict_bootstraps_results2,
                                    dict_bootstraps_results3)
