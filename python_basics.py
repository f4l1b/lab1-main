"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    raise NotImplementedError  # TODO


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    raise NotImplementedError  # TODO


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    raise NotImplementedError  # TODO


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    raise NotImplementedError  # TODO


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    raise NotImplementedError  # TODO


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    raise NotImplementedError  # TODO


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    raise NotImplementedError  # TODO


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    raise NotImplementedError  # TODO

def count_vowels(s):
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in s if ch in vowels)

def has_unique_characters(s: str) -> bool:
    seen = set()
    for ch in s:
        if ch in seen:
            return False
        seen.add(ch)
    return True

def count_one_bits(n: int) -> int:
    return bin(n).count('1')

def multiplicative_persistence(n):
    count = 0
    while n >= 10:
        product = 1
        while n > 0:
            product *= n % 10
            n //= 10
        n = product
        count += 1
    return count

def mse(y_true, y_pred):
    n = len(y_true)
    total_squared_error = sum((x - y) ** 2 for x, y in zip(y_true, y_pred))
    return total_squared_error / n

def prime_factorization(n):
    factors = []
    divisor = 2
    
    while divisor * divisor <= n:
        if n % divisor == 0:
            power = 0
            while n % divisor == 0:
                n //= divisor
                power += 1
            factors.append((divisor, power))
        divisor += 1
    
    if n > 1:
        factors.append((n, 1))
    
    result = ""
    for prime, power in factors:
        if power == 1:
            result += f"({prime})"
        else:
            result += f"({prime}**{power})"
            
    return result

def pyramid(n):
    total = 0
    k = 1
    
    while total < n:
        total += k ** 2
        k += 1
        
    if total == n:
        return k - 1
    else:
        return "It is impossible"

def is_balanced_number(n):
    s = str(n)
    length = len(s)
    
    if length % 2 == 0:
        left_part = s[:length // 2 - 1]
        right_part = s[length // 2 + 1:]
    else:
        left_part = s[:length // 2]
        right_part = s[length // 2 + 1:]
        
    sum_left = sum(int(d) for d in left_part)
    sum_right = sum(int(d) for d in right_part)
    
    return sum_left == sum_right

