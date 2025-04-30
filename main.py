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
        self.next_offset = 0
    
    def declare(self, key, var_type):
        if key in self.symbols:
            raise ValueError(f'Variável "{key}" já declarada.')
        size = 4
        self.next_offset += size
        self.symbols[key] = {"type": var_type, "offset": -self.next_offset}
    
    def set(self, key, val, val_type):
        if key not in self.symbols:
            raise ValueError(f'Variável não declarada: {key}')
        expected_type = self.symbols[key]["type"]
        if expected_type != val_type:
            raise ValueError(f'Tentando atribuir {val_type} em variável do tipo {expected_type}')
        self.symbols[key]["value"] = val
    
    def get(self, key):
        if key not in self.symbols:
            raise ValueError(f'Variável não declarada: {key}')
        return (self.symbols[key]["offset"], self.symbols[key]["type"])

    def get_offset(self, key):
        return self.symbols[key]["offset"]

class Code:
    def __init__(self):
        self.instructions = []

    def add(self, instr):
        self.instructions.append(instr)

    def write(self, source_file):
        out_file = source_file.rsplit('.',1)[0] + '.asm'
        with open(out_file, 'w') as f:
            f.write("section .text\n")
            f.write("_start:\n")
            for instr in self.instructions:
                f.write(f"    {instr}\n")
            f.write("    mov eax, 1\n")
            f.write("    mov ebx, 0\n")
            f.write("    int 0x80\n")
        print(f"Gerado: {out_file}")

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
            "var": "VAR",    
            "int": "INT_TYPE",
            "bool": "BOOL_TYPE",
            "string": "STRING_TYPE",
        }

    def selectNext(self):
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        
        if self.position >= len(self.source):
            self.next = Token('EOF', None)
            return
        
        current_char = self.source[self.position]

        if current_char == '"':
            start = self.position + 1
            self.position += 1
            string_content = []
            while self.position < len(self.source) and self.source[self.position] != '"':
                string_content.append(self.source[self.position])
                self.position += 1
            if self.position >= len(self.source):
                raise ValueError("String não fechada com aspas")
            self.position += 1
            self.next = Token('STRING', "".join(string_content))
            return

        if current_char.isdigit():
            start = self.position
            while self.position < len(self.source) and self.source[self.position].isdigit():
                self.position += 1
            number_str = self.source[start:self.position]
            self.next = Token('NUMBER', int(number_str))
            return
        
        if current_char.isalpha():
            start = self.position
            while (self.position < len(self.source)
                   and (self.source[self.position].isalnum() or self.source[self.position] == '_')):
                self.position += 1
            identifier = self.source[start:self.position]
            if identifier in self.keywords:
                self.next = Token(self.keywords[identifier], None)
            else:
                self.next = Token('IDENTIFIER', identifier)
            return
        
        if current_char == '+':
            self.position += 1
            self.next = Token('PLUS', None)
            return
        if current_char == '-':
            self.position += 1
            self.next = Token('MINUS', None)
            return
        if current_char == '*':
            self.position += 1
            self.next = Token('MULTIPLY', None)
            return
        if current_char == '/':
            self.position += 1
            self.next = Token('DIVIDE', None)
            return
        if current_char == '(':
            self.position += 1
            self.next = Token('LPAREN', None)
            return
        if current_char == ')':
            self.position += 1
            self.next = Token('RPAREN', None)
            return
        if current_char == '{':
            self.position += 1
            self.next = Token('LBRACE', None)
            return
        if current_char == '}':
            self.position += 1
            self.next = Token('RBRACE', None)
            return
        if current_char == '=':
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
                self.position += 2
                self.next = Token('EQUALS', None)
            else:
                self.position += 1
                self.next = Token('ASSIGN', None)
            return
        if current_char == '>':
            self.position += 1
            self.next = Token('GREATER', None)
            return
        if current_char == '<':
            self.position += 1
            self.next = Token('LESS', None)
            return
        if current_char == '&' and (self.position + 1 < len(self.source) and self.source[self.position+1] == '&'):
            self.position += 2
            self.next = Token('AND', None)
            return
        if current_char == '|' and (self.position + 1 < len(self.source) and self.source[self.position+1] == '|'):
            self.position += 2
            self.next = Token('OR', None)
            return
        if current_char == '!':
            self.position += 1
            self.next = Token('NOT', None)
            return
        
        raise ValueError(f'Caractere inválido: {current_char}')

class Node:
    id = 0

    @staticmethod
    def newId():
        Node.id += 1
        return Node.id

    def __init__(self):
        self.value = None
        self.children = []
        self.id = Node.newId()

    def evaluate(self):
        pass

    def generate(self, symbol_table, code):
        raise NotImplementedError(f"Generate não implementado para {type(self)}")

class If(Node):
    def __init__(self, condition, then_block, else_block=None):
        super().__init__()
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

    def evaluate(self, symbol_table):
        cond_val, cond_type = self.condition.evaluate(symbol_table)
        if cond_type != 'bool':
            raise ValueError("Condição do if deve ser bool")
        
        if cond_val:
            return self.then_block.evaluate(symbol_table)
        elif self.else_block:
            return self.else_block.evaluate(symbol_table)
        return (None, None)

    def generate(self, symbol_table, code):
        end_lbl=f'end_{self.id}' 
        else_lbl=f'else_{self.id}'
        self.condition.generate(symbol_table, code)
        code.add('cmp eax,0')
        code.add(f'je {else_lbl}')
        self.then_block.generate(symbol_table, code)
        code.add(f'jmp {end_lbl}')
        code.add(f'{else_lbl}:')
        if self.else_block: 
            self.else_block.generate(symbol_table, code)
        code.add(f'{end_lbl}:')

class For(Node):
    def __init__(self, init, condition, increment, block):
        super().__init__()
        self.init = init
        self.condition = condition
        self.increment = increment
        self.block = block

    def evaluate(self, symbol_table):
        self.init.evaluate(symbol_table)
        while True:
            cond_val, cond_type = self.condition.evaluate(symbol_table)
            if cond_type != 'bool':
                raise ValueError("Condição do for deve ser bool")
            if not cond_val:
                break
            self.block.evaluate(symbol_table)
            self.increment.evaluate(symbol_table)
        return (None, None)

    def generate(self, symbol_table, code):
        start_lbl = f'start_{self.id}'
        end_lbl = f'end_{self.id}'
        self.init.generate(symbol_table, code)
        code.add(f'{start_lbl}:')
        self.condition.generate(symbol_table, code)
        code.add('cmp eax,0')
        code.add(f'je {end_lbl}')
        self.block.generate(symbol_table, code)
        self.increment.generate(symbol_table, code)
        code.add(f'jmp {start_lbl}')
        code.add(f'{end_lbl}:')

class Scan(Node):
    def __init__(self):
        super().__init__()
        
    def evaluate(self, symbol_table):
        input_value = input().strip()
        try:
            val_int = int(input_value)
            return (val_int, 'int')
        except ValueError:
            return (input_value, 'string')
    
    def generate(self, symbol_table, code):
        code.add("; call scanf")

class BinOp(Node):
    def __init__(self, value, left, right):
        super().__init__()
        self.value = value
        self.children = [left, right]

    def evaluate(self, symbol_table):
        left_val, left_type = self.children[0].evaluate(symbol_table)
        right_val, right_type = self.children[1].evaluate(symbol_table)

        op = self.value

        if op in ['==', '>', '<']:
            if left_type != right_type:
                raise ValueError(f"Não é possível comparar tipos diferentes: {left_type} e {right_type}")
            
            if op == '==':
                return (left_val == right_val, 'bool')
            if op == '>':
                if left_type == 'int' and right_type == 'int':
                    return (left_val > right_val, 'bool')
                elif left_type == 'string' and right_type == 'string':
                    return (left_val > right_val, 'bool')
                else:
                    raise ValueError(f'Operador ">" só pode ser usado entre ints ou strings')
            if op == '<':
                if left_type == 'int' and right_type == 'int':
                    return (left_val < right_val, 'bool')
                elif left_type == 'string' and right_type == 'string':
                    return (left_val < right_val, 'bool')
                else:
                    raise ValueError(f'Operador "<" só pode ser usado entre ints ou strings')
                
        if op in ['&&', '||']:
            if left_type != 'bool' or right_type != 'bool':
                raise ValueError(f"Operador {op} requer bool e bool")
            if op == '&&':
                return (left_val and right_val, 'bool')
            else: 
                return (left_val or right_val, 'bool')
        if op == '+':
            if left_type == 'int' and right_type == 'int':
                return (left_val + right_val, 'int')
            elif left_type == 'string' or right_type == 'string':
                if left_type == 'bool':
                    left_val = 'true' if left_val else 'false'
                else:
                    left_val = str(left_val)
                if right_type == 'bool':
                    right_val = 'true' if right_val else 'false'
                else:
                    right_val = str(right_val)
                return (left_val + right_val, 'string')
            else:
                raise ValueError(f"Operador + não suportado para {left_type} e {right_type}")
        if op == '-':
            if left_type == 'int' and right_type == 'int':
                return (left_val - right_val, 'int')
            else:
                raise ValueError(f"Operador - não suportado para {left_type} e {right_type}")
        if op == '*':
            if left_type == 'int' and right_type == 'int':
                return (left_val * right_val, 'int')
            else:
                raise ValueError(f"Operador * não suportado para {left_type} e {right_type}")
        if op == '/':
            if left_type == 'int' and right_type == 'int':
                if right_val == 0:
                    raise ValueError('Divisão por zero')
                return (left_val // right_val, 'int')
            else:
                raise ValueError(f"Operador / não suportado para {left_type} e {right_type}")

        raise ValueError(f"Operador desconhecido: {op}")

    def generate(self, symbol_table, code):
        # left -> EAX
        self.children[0].generate(symbol_table, code)
        code.add("push eax    ; salva L")
        # right -> EAX
        self.children[1].generate(symbol_table, code)
        code.add("mov ecx, eax    ; R em ECX")
        code.add("pop eax        ; L em EAX")
        op = self.value
        if op == '+':
            code.add("add eax, ecx")
        elif op == '-':
            code.add("sub eax, ecx")
        elif op == '*':
            code.add("imul eax, ecx")
        elif op == '/':
            code.add("xor edx, edx")
            code.add("idiv ecx")
        elif op == '==':
            code.add("cmp eax, ecx")
            code.add("sete al")
            code.add("movzx eax, al")
        elif op == '<':
            code.add("cmp eax, ecx")
            code.add("setl al")
            code.add("movzx eax, al")
        elif op == '>':
            code.add("cmp eax, ecx")
            code.add("setg al")
            code.add("movzx eax, al")
        elif op == '&&':
            code.add("cmp eax, 0")
            code.add("je false")
            code.add("cmp ecx, 0")
            code.add("je false")
            code.add("mov eax, 1")
            code.add("jmp end")
            code.add("false:")
            code.add("mov eax, 0")
            code.add("end:")
        elif op == '||':
            code.add("cmp eax, 0")
            code.add("jne true")
            code.add("cmp ecx, 0")
            code.add("jne true")
            code.add("mov eax, 0")
            code.add("jmp end")
            code.add("true:")
            code.add("mov eax, 1")
            code.add("end:")
        else:
            raise ValueError(f"Operador desconhecido: {op}")

class UnOp(Node):
    def __init__(self, value, child):
        super().__init__()
        self.value = value
        self.children = [child]

    def evaluate(self, symbol_table):
        child_val, child_type = self.children[0].evaluate(symbol_table)
        op = self.value
        if op == '+':
            if child_type != 'int':
                raise ValueError(f"Operador unário + não aplicável em {child_type}")
            return (child_val, 'int')
        elif op == '-':
            if child_type != 'int':
                raise ValueError(f"Operador unário - não aplicável em {child_type}")
            return (-child_val, 'int')
        elif op == '!':
            if child_type != 'bool':
                raise ValueError(f"Operador ! requer bool")
            return (not child_val, 'bool')
        else:
            raise ValueError(f"Operador unário desconhecido: {op}")
    
    def generate(self, symbol_table, code):
        self.children[0].generate(symbol_table, code)
        op = self.value
        if op == '+':
            pass
        elif op == '-':
            code.add("neg eax")
        elif op == '!':
            code.add("cmp eax, 0")
            code.add("sete al")
            code.add("movzx eax, al")
        else:
            raise ValueError(f"Operador unário desconhecido: {op}")

class IntVal(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, symbol_table):
        return (self.value, 'int')

    def generate(self, symbol_table, code):
        code.add(f"mov eax, {self.value}")

class StrVal(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, symbol_table):
        return (self.value, 'string')
    
    def generate(self, symbol_table, code): 
        code.add(f"; carregar string '{self.value}' em eax")

class BoolVal(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, symbol_table):
        return (self.value, 'bool')

    def generate(self, symbol_table, code):
        if self.value:
            code.add("mov eax, 1")
        else:
            code.add("mov eax, 0")

class NoOp(Node):
    def __init__(self):
        super().__init__()

    def evaluate(self, symbol_table):
        return (None, None)

    def generate(self, symbol_table, code):
        pass

class Block(Node):
    def __init__(self, statements):
        self.children = statements
    
    def evaluate(self, symbol_table):
        ret = (None, None)
        for statement in self.children:
            ret = statement.evaluate(symbol_table)
        return ret

    def generate(self, symbol_table, code):
        for statement in self.children:
            statement.generate(symbol_table, code)

class Println(Node):
    def __init__(self, expression):
        self.children = [expression]
    
    def evaluate(self, symbol_table):
        val, val_type = self.children[0].evaluate(symbol_table)
        if val is None:
            print("None")
        elif val_type == 'bool':
            print("true" if val else "false")
        else:
            print(val)
        return (val, val_type)

    def generate(self, symbol_table, code):
        self.children[0].generate(symbol_table, code)
        code.add("push eax")
        code.add("call print_int")
        code.add("add esp, 4")

class Identifier(Node):
    def __init__(self, name):
        super().__init__()
        self.value = name
    
    def evaluate(self, symbol_table):
        return symbol_table.get(self.value)
    
    def generate(self, symbol_table, code):
        off,_=symbol_table.get(self.value); code.add(f'mov eax,[ebp{off:+}]')

class Assignment(Node):
    def __init__(self, identifier, expression):
        super().__init__()
        self.children = [identifier, expression]
    
    def evaluate(self, symbol_table):
        expr_val, expr_type = self.children[1].evaluate(symbol_table)
        var_name = self.children[0].value
        symbol_table.set(var_name, expr_val, expr_type)
        return (expr_val, expr_type)
    

    def generate(self, symbol_table, code):
        self.children[1].generate(symbol_table, code)         
        offset, _ = symbol_table.get(self.children[0].value)
        code.add(f"mov [ebp{offset:+}], eax    ; atribui {self.children[0].value}")

class VarDecl(Node):
    def __init__(self, identifier, var_type, expression=None):
        super().__init__()
        self.identifier = identifier
        self.var_type = var_type
        self.expression = expression

    def evaluate(self, symbol_table):
        symbol_table.declare(self.identifier, self.var_type)
        if self.expression is not None:
            val, val_type = self.expression.evaluate(symbol_table)
            if val_type != self.var_type:
                raise ValueError(f"Incompatibilidade de tipo ao inicializar '{self.identifier}'. Esperado {self.var_type}, obteve {val_type}")
            symbol_table.set(self.identifier, val, val_type)
            return (val, val_type)
        return (None, None)
    
    def generate(self, symbol_table, code):
        symbol_table.declare(self.identifier, self.var_type)
        code.add(f"sub esp, 4    ; aloca {self.identifier}")
        if self.expression:
            self.expression.generate(symbol_table, code)
            offset = symbol_table.get_offset(self.identifier)
            code.add(f"mov [ebp{offset:+}], eax    ; inicializa {self.identifier}")

class Parser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.selectNext()

    def parseStatement(self):
        # var x int [= expr]
        if self.tokenizer.next.type == 'VAR':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'IDENTIFIER':
                raise ValueError("Esperado identificador após 'var'")
            var_name = self.tokenizer.next.value
            self.tokenizer.selectNext()

            if self.tokenizer.next.type not in ('INT_TYPE','BOOL_TYPE','STRING_TYPE'):
                raise ValueError("Esperado tipo após nome da variável. Use: int, bool ou string.")
            if self.tokenizer.next.type == 'INT_TYPE':
                var_type = 'int'
            elif self.tokenizer.next.type == 'BOOL_TYPE':
                var_type = 'bool'
            else:
                var_type = 'string'
            self.tokenizer.selectNext()

            expr = None
            if self.tokenizer.next.type == 'ASSIGN':
                self.tokenizer.selectNext()
                expr = self.parseBExpression()
            
                if self.tokenizer.next.type not in (
                    'LBRACE', 'RBRACE', 'IF', 'FOR', 
                    'PRINTLN', 'IDENTIFIER', 'EOF', 'VAR'
                ):
                    raise ValueError(f"Token inesperado após expressão de atribuição: {self.tokenizer.next}")
            
            return VarDecl(var_name, var_type, expr)
        
        # Instrução println: Println(expr)
        if self.tokenizer.next.type == 'PRINTLN':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'LPAREN':
                raise ValueError('Parêntese de abertura esperado')
            self.tokenizer.selectNext()
            expression = self.parseBExpression()
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
                if (self.tokenizer.next.type == 'LBRACE' 
                    and self.tokenizer.next.type != 'RBRACE'):
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
            expression = self.parseBExpression()
            return Assignment(identifier, expression)
        
        # If
        if self.tokenizer.next.type == 'IF':
            self.tokenizer.selectNext()
            condition = self.parseBExpression()
            if self.tokenizer.next.type != 'LBRACE':
                raise ValueError('Esperado { após if')
            then_block = self.parseStatement()
            if isinstance(then_block, Block) and not then_block.children:
                raise ValueError("Bloco do if está vazio")
            else_block = None
            if self.tokenizer.next.type == 'ELSE':
                self.tokenizer.selectNext()
                if self.tokenizer.next.type != 'LBRACE':
                    raise ValueError('Esperado { após else')
                else_block = self.parseStatement()
                if isinstance(else_block, Block) and not else_block.children:
                    raise ValueError("Bloco do else está vazio")
            return If(condition, then_block, else_block)
        
        # Loop for
        if self.tokenizer.next.type == 'FOR':
            self.tokenizer.selectNext()
            condition = self.parseBExpression()
            if self.tokenizer.next.type != 'LBRACE':
                raise ValueError('Esperado { imediatamente após condição do for')
            block = self.parseStatement()
            if isinstance(block, Block) and not block.children:
                raise ValueError("Bloco do for está vazio")
            return For(NoOp(), condition, NoOp(), block)
        
        else:
            return self.parseBExpression()

    def parseBExpression(self):
        result = self.parseBTerm()
        while self.tokenizer.next.type == 'OR':
            self.tokenizer.selectNext()
            right = self.parseBTerm()
            result = BinOp('||', result, right)
        return result
    
    def parseBTerm(self):
        result = self.parseRelExpression()
        while self.tokenizer.next.type == 'AND':
            self.tokenizer.selectNext()
            right = self.parseRelExpression()
            result = BinOp('&&', result, right)
        return result

    def parseRelExpression(self):
        result = self.parseExpression()
        while self.tokenizer.next.type in ('LESS', 'GREATER', 'EQUALS'):
            if self.tokenizer.next.type == 'LESS':
                self.tokenizer.selectNext()
                right = self.parseExpression()
                result = BinOp('<', result, right)
            elif self.tokenizer.next.type == 'GREATER':
                self.tokenizer.selectNext()
                right = self.parseExpression()
                result = BinOp('>', result, right)
            elif self.tokenizer.next.type == 'EQUALS':
                self.tokenizer.selectNext()
                right = self.parseExpression()
                result = BinOp('==', result, right)
        return result

    def parseExpression(self):
        result = self.parseTerm()
        while self.tokenizer.next.type in ('PLUS', 'MINUS'):
            if self.tokenizer.next.type == 'PLUS':
                self.tokenizer.selectNext()
                right = self.parseTerm()
                result = BinOp('+', result, right)
            elif self.tokenizer.next.type == 'MINUS':
                self.tokenizer.selectNext()
                right = self.parseTerm()
                result = BinOp('-', result, right)
        return result

    def parseTerm(self):
        result = self.parseFactor()
        while self.tokenizer.next.type in ('MULTIPLY', 'DIVIDE'):
            if self.tokenizer.next.type == 'MULTIPLY':
                self.tokenizer.selectNext()
                right = self.parseFactor()
                result = BinOp('*', result, right)
            elif self.tokenizer.next.type == 'DIVIDE':
                self.tokenizer.selectNext()
                right = self.parseFactor()
                result = BinOp('/', result, right)
        return result

    def parseFactor(self):
        if self.tokenizer.next.type == 'NUMBER':
            result = IntVal(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            return result
        elif self.tokenizer.next.type == 'STRING':
            result = StrVal(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            return result
        elif self.tokenizer.next.type == 'LPAREN':
            self.tokenizer.selectNext()
            result = self.parseBExpression()
            if self.tokenizer.next.type != 'RPAREN':
                raise ValueError('Parêntese de fechamento esperado')
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
        elif self.tokenizer.next.type == 'IDENTIFIER':
            result = Identifier(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            return result
        elif self.tokenizer.next.type == 'SCAN':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'LPAREN':
                raise ValueError('Parêntese de abertura esperado após Scan')
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'RPAREN':
                raise ValueError('Parêntese de fechamento esperado')
            self.tokenizer.selectNext()
            return Scan()
        else:
            raise ValueError(f"Fator inválido ou não suportado: {self.tokenizer.next.type}")

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

        code = Code()
        ast.generate(symbol_table, code)
        code.write(file)

        # result = ast.evaluate(symbol_table)
        # return result
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