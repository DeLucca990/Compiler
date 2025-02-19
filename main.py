class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

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
        else:
            raise ValueError(f'Caractere inválido: {current_char}')

class Parser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def parseExpression(self):
        left = self.parseTerm()
        while self.tokenizer.next.type in ('PLUS', 'MINUS'):
            op = self.tokenizer.next
            self.tokenizer.selectNext()
            right = self.parseTerm()
            if op.type == 'PLUS':
                left += right
            elif op.type == 'MINUS':
                left -= right
        return left
    
    def parseTerm(self):
        if self.tokenizer.next.type != 'NUMBER':
            raise ValueError('Número esperado, mas não encontrado')
        left = self.tokenizer.next.value
        self.tokenizer.selectNext()
        while self.tokenizer.next.type in ('MULTIPLY', 'DIVIDE'):
            op = self.tokenizer.next
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'NUMBER':
                raise ValueError('Número esperado, mas não encontrado')
            right = self.tokenizer.next.value
            self.tokenizer.selectNext()
            if op.type == 'MULTIPLY':
                left *= right
            elif op.type == 'DIVIDE':
                if right == 0:
                    raise ValueError('Divisão por zero')
                left //= right
        return left

    @staticmethod
    def run(code):
        tokenizer = Tokenizer(code)
        tokenizer.selectNext()
        parser = Parser(tokenizer)
        result = parser.parseExpression()

        if tokenizer.next.type != "EOF":
            raise ValueError(f'EOF esperado, mas obtido: {tokenizer.next}')
        return result

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('python3 main.py <expressão>')
        raise ValueError('Argumentos inválidos')

    code = sys.argv[1]

    if code == "":
        raise ValueError('Expressão inválida')

    print(Parser.run(code))