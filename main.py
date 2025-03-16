import sys
import re

class PrePro:
    @staticmethod
    def filter(code):
        code = re.sub(r'\/\/.*', '', code)
        return code

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    
    def __str__(self):
        return f"Token({self.type}, {self.value})"

class Tokenizer:
    def __init__(self, source):
        self.source = source
        self.position = 0
        self.next = None

    def selectNext(self):
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        
        if self.position >= len(self.source):
            self.next = Token('EOF', None)
            return
        
        current_char = self.source[self.position]
        if current_char.isdigit():
            start = self.position
            while self.position < len(self.source) and self.source[self.position].isdigit():
                self.position += 1
            number_str = self.source[start:self.position]
            self.next = Token('NUMBER', int(number_str))
        elif current_char == '+':
            self.next = Token('PLUS', None)
            self.position += 1
        elif current_char == '-':
            self.next = Token('MINUS', None)
            self.position += 1
        elif current_char == '*':
            self.next = Token('MULTIPLY', None)
            self.position += 1
        elif current_char == '/':
            self.next = Token('DIVIDE', None)
            self.position += 1
        elif current_char == '(':
            self.next = Token('LPAREN', None)
            self.position += 1
        elif current_char == ')':
            self.next = Token('RPAREN', None)
            self.position += 1
        else:
            raise ValueError(f'Caractere inválido: {current_char}')

class Node:
    def __init__(self):
        self.value = None
        self.children = []

    def evaluate(self):
        pass

class BinOp(Node):
    def __init__(self, value, left, right):
        super().__init__()
        self.value = value
        self.children = [left, right]

    def evaluate(self):
        if self.value == '+':
            return self.children[0].evaluate() + self.children[1].evaluate()
        elif self.value == '-':
            return self.children[0].evaluate() - self.children[1].evaluate()
        elif self.value == '*':
            return self.children[0].evaluate() * self.children[1].evaluate()
        elif self.value == '/':
            divisor = self.children[1].evaluate()
            if divisor == 0:
                raise ValueError('Divisão por zero')
            return self.children[0].evaluate() // divisor

class UnOp(Node):
    def __init__(self, value, child):
        super().__init__()
        self.value = value
        self.children = [child]

    def evaluate(self):
        if self.value == '+':
            return self.children[0].evaluate()
        elif self.value == '-':
            return -self.children[0].evaluate()

class IntVal(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self):
        return self.value

class NoOp(Node):
    def __init__(self):
        super().__init__()

    def evaluate(self):
        return 0

class Parser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.selectNext()
    
    def parseExpression(self):
        result = self.parseTerm()
        while self.tokenizer.next.type in ('PLUS', 'MINUS'):
            if self.tokenizer.next.type == 'PLUS':
                self.tokenizer.selectNext()
                result = BinOp('+', result, self.parseTerm())
            elif self.tokenizer.next.type == 'MINUS':
                self.tokenizer.selectNext()
                result = BinOp('-', result, self.parseTerm())
        return result
    
    def parseTerm(self):
        result = self.parseFactor()
        while self.tokenizer.next.type in ('MULTIPLY', 'DIVIDE'):
            if self.tokenizer.next.type == 'MULTIPLY':
                self.tokenizer.selectNext()
                result = BinOp('*', result, self.parseFactor())
            elif self.tokenizer.next.type == 'DIVIDE':
                self.tokenizer.selectNext()
                result = BinOp('/', result, self.parseFactor())
        return result

    def parseFactor(self):
        if self.tokenizer.next.type == 'NUMBER':
            result = IntVal(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            return result
        elif self.tokenizer.next.type == 'PLUS':
            self.tokenizer.selectNext()
            return UnOp('+', self.parseFactor())
        elif self.tokenizer.next.type == 'MINUS':
            self.tokenizer.selectNext()
            return UnOp('-', self.parseFactor())
        elif self.tokenizer.next.type == 'LPAREN':
            self.tokenizer.selectNext()
            result = self.parseExpression()
            if self.tokenizer.next.type != 'RPAREN':
                raise ValueError('Parêntese de fechamento esperado')
            self.tokenizer.selectNext()
            return result
        else:
            raise ValueError("Fator inválido")

    def parse(self):
        result = self.parseExpression()
        if self.tokenizer.next.type != 'EOF':
            raise ValueError(f'EOF esperado, mas obtido: {self.tokenizer.next}')
        return result

def main(file):
    try:
        with open(file, 'r') as f:
            code = f.read()
        
        filtered_code = PrePro.filter(code)

        if not filtered_code.strip():
            raise ValueError('Código vazio')
        
        tokenizer = Tokenizer(filtered_code)
        parser = Parser(tokenizer)

        ast = parser.parse()

        result = ast.evaluate()

        print(result)
        return result
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file}' não encontrado.")
        sys.exit(1)
    except Exception as e:
        raise e

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python3 main.py <arquivo>")
        sys.exit(1)
    
    file = sys.argv[1]
    main(file)