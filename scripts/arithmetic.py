"""Compile arithmetic equalities into exact computations for individual symbols."""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import prod


class EquationError(ValueError):
    pass


Monomial = tuple[str, ...]
Polynomial = dict[Monomial, Fraction]
ONE: Polynomial = {(): Fraction(1)}
SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
TOKEN = re.compile(rf"\s+|{NUMBER.pattern}|{SYMBOL.pattern}|[()+*/-]")


def _scale(polynomial: Polynomial, factor: Fraction) -> Polynomial:
    return {term: coefficient * factor for term, coefficient in polynomial.items()} if factor else {}


def _add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = left.copy()
    for term, coefficient in right.items():
        result[term] = result.get(term, Fraction(0)) + coefficient
    return {term: coefficient for term, coefficient in result.items() if coefficient}


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result = defaultdict(Fraction)
    for left_term, left_coefficient in left.items():
        for right_term, right_coefficient in right.items():
            term = tuple(sorted(left_term + right_term))
            result[term] += left_coefficient * right_coefficient
    return {term: coefficient for term, coefficient in result.items() if coefficient}


def _exact_quotient(polynomial: Polynomial, divisor: Polynomial) -> Polynomial | None:
    leading_divisor = max(divisor, key=lambda term: (len(term), term))
    divisor_powers = Counter(leading_divisor)
    remainder = polynomial
    quotient = {}
    while remainder:
        leading = max(remainder, key=lambda term: (len(term), term))
        powers = Counter(leading)
        if not divisor_powers <= powers:
            return None
        term = tuple(sorted((powers - divisor_powers).elements()))
        coefficient = remainder[leading] / divisor[leading_divisor]
        quotient[term] = coefficient
        remainder = _add(remainder, _multiply(divisor, {term: -coefficient}))
    return quotient


def _cancel_nonzero_factors(polynomial: Polynomial, conditions: tuple[Polynomial, ...]) -> Polynomial:
    for condition in conditions:
        if not any(condition):
            continue
        # A nonzero product also makes every symbol common to its terms nonzero.
        symbols = set.intersection(*(set(term) for term in condition))
        factors = [condition] + [{(name,): Fraction(1)} for name in sorted(symbols)]
        for factor in factors:
            while polynomial:
                quotient = _exact_quotient(polynomial, factor)
                if quotient is None:
                    break
                polynomial = quotient
    return polynomial


@dataclass(frozen=True)
class _Rational:
    numerator: Polynomial
    denominator: Polynomial
    nonzero: tuple[Polynomial, ...] = ()

    @classmethod
    def simplified(cls, numerator, denominator, nonzero):
        if not numerator or numerator == denominator:
            return cls(ONE if numerator else {}, ONE, nonzero)
        if set(denominator) == {()}:
            return cls(_scale(numerator, 1 / denominator[()]), ONE, nonzero)
        return cls(numerator, denominator, nonzero)

    def __add__(self, other):
        if self.denominator == other.denominator:
            numerator = _add(self.numerator, other.numerator)
            denominator = self.denominator
        else:
            numerator = _add(
                _multiply(self.numerator, other.denominator),
                _multiply(other.numerator, self.denominator),
            )
            denominator = _multiply(self.denominator, other.denominator)
        return self.simplified(numerator, denominator, self.nonzero + other.nonzero)

    def __neg__(self):
        return _Rational(_scale(self.numerator, Fraction(-1)), self.denominator, self.nonzero)

    def __mul__(self, other):
        return self.simplified(
            _multiply(self.numerator, other.numerator),
            _multiply(self.denominator, other.denominator),
            self.nonzero + other.nonzero,
        )

    def __truediv__(self, other):
        if not other.numerator:
            raise EquationError("Division by zero in equation")
        return self.simplified(
            _multiply(self.numerator, other.denominator),
            _multiply(self.denominator, other.numerator),
            self.nonzero + other.nonzero + (other.numerator,),
        )


class _Parser:
    def __init__(self, text: str):
        self.tokens = []
        position = 0
        while position < len(text):
            match = TOKEN.match(text, position)
            if match is None:
                raise EquationError(f"Unexpected character {text[position]!r}")
            token = match.group()
            if not token.isspace():
                self.tokens.append(token)
            position = match.end()
        self.tokens.append("")
        self.position = 0

    @property
    def next(self):
        return self.tokens[self.position]

    def take(self):
        token = self.next
        if not token:
            raise EquationError("Expected a number, variable, or opening parenthesis")
        self.position += 1
        return token

    def parse(self):
        value = self.sum()
        if self.next:
            raise EquationError(f"Unexpected token {self.next!r}")
        return value

    def sum(self):
        value = self.product()
        while self.next in ("+", "-"):
            operator = self.take()
            right = self.product()
            value = value + (right if operator == "+" else -right)
        return value

    def product(self):
        value = self.factor()
        while self.next in ("*", "/", "(") or SYMBOL.fullmatch(self.next):
            operator = self.take() if self.next in ("*", "/") else "*"
            right = self.factor()
            value = value * right if operator == "*" else value / right
        return value

    def factor(self):
        token = self.take()
        if token in ("+", "-"):
            value = self.factor()
            return value if token == "+" else -value
        if token == "(":
            value = self.sum()
            if self.next != ")":
                raise EquationError("Expected a closing parenthesis")
            self.take()
            return value
        if NUMBER.fullmatch(token):
            coefficient = Fraction(token)
            return _Rational({(): coefficient} if coefficient else {}, ONE)
        if SYMBOL.fullmatch(token):
            return _Rational({(token,): Fraction(1)}, ONE)
        raise EquationError(f"Expected a number, variable, or opening parenthesis, got {token!r}")


def _evaluate(polynomial: Polynomial, values: dict[str, Fraction]) -> Fraction:
    return sum(
        (coefficient * prod(values[name] for name in term) for term, coefficient in polynomial.items()),
        Fraction(0),
    )


def _pack(polynomial):
    return [[term, str(coefficient)] for term, coefficient in sorted(polynomial.items())]


def _unpack(rows):
    return {tuple(term): Fraction(coefficient) for term, coefficient in rows}


@dataclass(frozen=True)
class Rule:
    target: str
    numerator: Polynomial
    denominator: Polynomial
    nonzero: tuple[Polynomial, ...]

    @property
    def inputs(self) -> tuple[str, ...]:
        variables = {
            name
            for polynomial in (self.numerator, self.denominator, *self.nonzero)
            for term in polynomial
            for name in term
        }
        return tuple(sorted(variables - {self.target}))

    def evaluate(self, values: dict[str, Fraction]) -> Fraction:
        denominator = _evaluate(self.denominator, values)
        if not denominator:
            raise EquationError(f"The coefficient of {self.target} must be nonzero")
        result = _evaluate(self.numerator, values) / denominator
        bindings = {**values, self.target: result}
        if any(not _evaluate(condition, bindings) for condition in self.nonzero):
            raise EquationError(f"A nonzero condition fails for {self.target} = {result}")
        return result

    def encode(self) -> str:
        return json.dumps([
            _pack(self.numerator), _pack(self.denominator),
            [_pack(condition) for condition in self.nonzero],
        ], separators=(",", ":"))

    @classmethod
    def decode(cls, target: str, data: str):
        numerator, denominator, conditions = json.loads(data)
        return cls(target, _unpack(numerator), _unpack(denominator), tuple(map(_unpack, conditions)))


def compile_value(expression: str) -> Rule:
    """Compile an expression whose variables are supplied as inputs."""
    if re.search(r"\b(?:floor|mod)\s*\(", expression):
        raise EquationError("Use floor and mod as top-level calculations")
    value = _Parser(expression).parse()
    return Rule("", value.numerator, value.denominator, value.nonzero)


def compile_equation(equation: str, targets: tuple[str, ...]) -> list[Rule]:
    left, equals, right = equation.partition("=")
    if not equals or "=" in right:
        raise EquationError("Expected one equality")
    difference = _Parser(left).parse() + -_Parser(right).parse()
    numerator = _cancel_nonzero_factors(difference.numerator, difference.nonzero)
    rules = []
    for target in targets:
        coefficient = {}
        constant = {}
        for term, value in numerator.items():
            power = term.count(target)
            if power > 1:
                raise EquationError(f"The equation must be linear in {target!r}")
            if power:
                coefficient[tuple(name for name in term if name != target)] = value
            else:
                constant[term] = value
        if not coefficient:
            raise EquationError(f"The equation does not determine {target!r}")
        conditions = []
        for condition in (*difference.nonzero, coefficient):
            if any(condition) and condition not in conditions:
                conditions.append(condition)
        solved = _Rational.simplified(_scale(constant, Fraction(-1)), coefficient, tuple(conditions))
        rules.append(Rule(target, solved.numerator, solved.denominator, solved.nonzero))
    return rules
