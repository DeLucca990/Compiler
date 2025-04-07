import sys
import re

class PrePro:
    @staticmethod
    def filter(code):
        code = re.sub(r'\/\/.*', '', code)
        code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)
        return code

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    
    def __str__(self):
        return f"Token({self.type}, {self.value})"

class SymbolTable:
    def __init__(self):
        self.symbols = {}
    def get(self, key):
        if key not in self.symbols:
            raise ValueError(f'Variável não declarada: {key}')
        return self.symbols[key]
    def set(self, key, value):
        self.symbols[key] = value

class Tokenizer:
    def __init__(self, source):
        self.source = source
        self.position = 0
        self.next = None
        self.keywords = {
            "Println": "PRINTLN",
            "if": "IF",
            "else": "ELSE",
            "for": "FOR",
            "Scan": "SCAN",
        }

    def selectNext(self):
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        
        if self.position >= len(self.source):
            self.next = Token('EOF', None)
            return
        
        current_char = self.source[self.position]
        
        # Se for dígito
        if current_char.isdigit():
            start = self.position
            while self.position < len(self.source) and self.source[self.position].isdigit():
                self.position += 1
            number_str = self.source[start:self.position]
            self.next = Token('NUMBER', int(number_str))
        # Se for letra
        elif current_char.isalpha():
            start = self.position
            while self.position < len(self.source) and (self.source[self.position].isalnum() or self.source[self.position] == '_'):
                self.position += 1
            identifier = self.source[start:self.position]
            if identifier in self.keywords:
                self.next = Token(self.keywords[identifier], None)
            else:
                self.next = Token('IDENTIFIER', identifier)
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
        elif current_char == '{':
            self.next = Token('LBRACE', None)
            self.position += 1
        elif current_char == '}':
            self.next = Token('RBRACE', None)
            self.position += 1
        elif current_char == '=':
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
                self.next = Token('EQUALS', None)
                self.position += 2
            else:
                self.next = Token('ASSIGN', None)
                self.position += 1
        elif current_char == '>':
            self.next = Token('GREATER', None)
            self.position += 1
        elif current_char == '<':
            self.next = Token('LESS', None)
            self.position += 1
        elif current_char == '&' and self.position + 1 < len(self.source) and self.source[self.position + 1] == '&':
            self.next = Token('AND', None)
            self.position += 2
        elif current_char == '|' and self.position + 1 < len(self.source) and self.source[self.position + 1] == '|':
            self.next = Token('OR', None)
            self.position += 2
        elif current_char == '!':
            self.next = Token('NOT', None)
            self.position += 1
        else:
            raise ValueError(f'Caractere inválido: {current_char}')

class Node:
    def __init__(self):
        self.value = None
        self.children = []

    def evaluate(self):
        pass

class If(Node):
    def __init__(self, condition, then_block, else_block=None):
        super().__init__()
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

    def evaluate(self, symbol_table):
        if self.condition.evaluate(symbol_table):
            return self.then_block.evaluate(symbol_table)
        elif self.else_block:
            return self.else_block.evaluate(symbol_table)

class For(Node):
    def __init__(self, init, condition, increment, block):
        super().__init__()
        self.init = init
        self.condition = condition
        self.increment = increment
        self.block = block

    def evaluate(self, symbol_table):
        self.init.evaluate(symbol_table)
        while self.condition.evaluate(symbol_table):
            self.block.evaluate(symbol_table)
            self.increment.evaluate(symbol_table)

class Scan(Node):
    def __init__(self):
        super().__init__()
    
    def evaluate(self, symbol_table):
        input_value = input()
        try:
            value = int(input_value)
            return value
        except ValueError:
            raise ValueError(f"Entrada inválida: '{input_value}' não é um número")

class BinOp(Node):
    def __init__(self, value, left, right):
        super().__init__()
        self.value = value
        self.children = [left, right]

    def evaluate(self, symbol_table):
        left_val = self.children[0].evaluate(symbol_table)
        right_val = self.children[1].evaluate(symbol_table)
        if self.value == '+':
            return left_val + right_val
        elif self.value == '-':
            return left_val - right_val
        elif self.value == '*':
            return left_val * right_val
        elif self.value == '/':
            if right_val == 0:
                raise ValueError('Divisão por zero')
            return left_val // right_val
        elif self.value == '==':
            return left_val == right_val
        elif self.value == '>':
            return left_val > right_val
        elif self.value == '<':
            return left_val < right_val
        elif self.value == '&&':
            return left_val and right_val
        elif self.value == '||':
            return left_val or right_val

class UnOp(Node):
    def __init__(self, value, child):
        super().__init__()
        self.value = value
        self.children = [child]

    def evaluate(self, symbol_table):
        child_val = self.children[0].evaluate(symbol_table)
        if self.value == '+':
            return child_val
        elif self.value == '-':
            return -child_val
        elif self.value == '!':
            return not child_val

class IntVal(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, symbol_table):
        return self.value

class NoOp(Node):
    def __init__(self):
        super().__init__()

    def evaluate(self, symbol_table):
        return 0

class Block(Node):
    def __init__(self, statements):
        self.children = statements
    
    def evaluate(self, symbol_table):
        for statement in self.children:
            statement.evaluate(symbol_table)

class Println(Node):
    def __init__(self, expression):
        self.children = [expression]
    
    def evaluate(self, symbol_table):
        result = self.children[0].evaluate(symbol_table)
        print(result)

class Identifier(Node):
    def __init__(self, name):
        super().__init__()
        self.value = name
    
    def evaluate(self, symbol_table):
        return symbol_table.get(self.value)

class Assignment(Node):
    def __init__(self, identifier, expression):
        super().__init__()
        self.children = [identifier, expression]
    
    def evaluate(self, symbol_table):
        value = self.children[1].evaluate(symbol_table)
        symbol_table.set(self.children[0].value, value)
        return value

class Parser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.selectNext()

    def parseStatement(self):
        # Instrução println: Println(expr)
        if self.tokenizer.next.type == 'PRINTLN':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'LPAREN':
                raise ValueError('Parêntese de abertura esperado')
            self.tokenizer.selectNext()
            expression = self.parseRelExpression()
            if self.tokenizer.next.type != 'RPAREN':
                raise ValueError('Parêntese de fechamento esperado')
            self.tokenizer.selectNext()
            return Println(expression)

        # Bloco de instruções {...}
        elif self.tokenizer.next.type == 'LBRACE':
            self.tokenizer.selectNext()
            statements = []
            while self.tokenizer.next.type != 'RBRACE':
                if self.tokenizer.next.type == 'LBRACE':
                    raise ValueError('Esperado } após o bloco')
                statement = self.parseStatement()
                statements.append(statement)
                if self.tokenizer.next.type == 'LBRACE' and self.tokenizer.next.type != 'RBRACE':
                    raise ValueError('Esperado } após o bloco')
            self.tokenizer.selectNext()
            return Block(statements)

        # Atribuição: identificador = expr
        elif self.tokenizer.next.type == 'IDENTIFIER':
            identifier = Identifier(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'ASSIGN':
                raise ValueError('Esperado = após identificador')
            self.tokenizer.selectNext()
            expression = self.parseRelExpression()
            if self.tokenizer.next.type not in ('LBRACE', 'RBRACE', 'IF', 'FOR', 'PRINTLN', 'IDENTIFIER', 'EOF'):
                raise ValueError(f"Unexpected token {self.tokenizer.next.type} (expected EOL)")
            return Assignment(identifier, expression)
        
        # If
        elif self.tokenizer.next.type == 'IF':
            self.tokenizer.selectNext()
            condition = self.parseRelExpression()
            then_block = self.parseStatement()
            else_block = None
            if self.tokenizer.next.type == 'ELSE':
                self.tokenizer.selectNext()
                else_block = self.parseStatement()
            return If(condition, then_block, else_block)
        
        # Loop for
        elif self.tokenizer.next.type == 'FOR':
            self.tokenizer.selectNext()
            condition = self.parseRelExpression()
            block = self.parseStatement()
            return For(NoOp(), condition, NoOp(), block)
        
        else:
            return self.parseRelExpression()

    def parseRelExpression(self):
        result = self.parseExpression()
        while self.tokenizer.next.type in ('LESS', 'GREATER', 'EQUALS', 'OR', 'AND'):
            if self.tokenizer.next.type == 'LESS':
                self.tokenizer.selectNext()
                result = BinOp('<', result, self.parseExpression())
            elif self.tokenizer.next.type == 'GREATER':
                self.tokenizer.selectNext()
                result = BinOp('>', result, self.parseExpression())
            elif self.tokenizer.next.type == 'EQUALS':
                self.tokenizer.selectNext()
                result = BinOp('==', result, self.parseExpression())
            elif self.tokenizer.next.type == 'OR':
                self.tokenizer.selectNext()
                result = BinOp('||', result, self.parseExpression())
            elif self.tokenizer.next.type == 'AND':
                self.tokenizer.selectNext()
                result = BinOp('&&', result, self.parseExpression())
        return result

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
        elif self.tokenizer.next.type == 'IDENTIFIER':
            result = Identifier(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            return result
        elif self.tokenizer.next.type == 'PLUS':
            self.tokenizer.selectNext()
            return UnOp('+', self.parseFactor())
        elif self.tokenizer.next.type == 'MINUS':
            self.tokenizer.selectNext()
            return UnOp('-', self.parseFactor())
        elif self.tokenizer.next.type == 'NOT':
            self.tokenizer.selectNext()
            return UnOp('!', self.parseFactor())
        elif self.tokenizer.next.type == 'SCAN':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'LPAREN':
                raise ValueError('Parêntese de abertura esperado após Scan')
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'RPAREN':
                raise ValueError('Parêntese de fechamento esperado')
            self.tokenizer.selectNext()
            return Scan()
        elif self.tokenizer.next.type == 'LPAREN':
            self.tokenizer.selectNext()
            result = self.parseRelExpression()
            if self.tokenizer.next.type != 'RPAREN':
                raise ValueError('Parêntese de fechamento esperado')
            self.tokenizer.selectNext()
            return result
        else:
            raise ValueError("Fator inválido")

    def parse(self):
        if self.tokenizer.next.type != 'LBRACE':
            raise ValueError('Esperado { no início do bloco')
        block = self.parseStatement()
        if self.tokenizer.next.type != 'EOF':
            raise ValueError('Esperado EOF após o bloco')
        return block

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

        symbol_table = SymbolTable()

        result = ast.evaluate(symbol_table)

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