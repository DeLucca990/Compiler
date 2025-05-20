import sys
import re
from typing import List, Tuple, Optional, Dict, Any

class PrePro:
    @staticmethod
    def filter(code: str) -> str:
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"/\*[\s\S]*?\*/", "", code)
        return code

class Token:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Tokenizer:
    KEYWORDS = {
        "Println": "PRINTLN",
        "Scan": "SCAN",
        "if": "IF",
        "else": "ELSE",
        "for": "FOR",
        "var": "VAR",
        "int": "INT_TYPE",
        "bool": "BOOL_TYPE",
        "string": "STRING_TYPE",
        "func": "FUNC",
        "return": "RETURN",
    }

    SIMPLE_TOKENS = {
        "+": "PLUS",
        "-": "MINUS",
        "*": "MULTIPLY",
        "/": "DIVIDE",
        "(": "LPAREN",
        ")": "RPAREN",
        "{": "LBRACE",
        "}": "RBRACE",
        ",": "COMMA",
    }

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.next: Token = None 
        self.select_next()

    def _advance_while(self, predicate):
        while self.position < len(self.source) and predicate(self.source[self.position]):
            self.position += 1

    def select_next(self):
        self._advance_while(str.isspace)
        if self.position >= len(self.source):
            self.next = Token("EOF", None)
            return

        ch = self.source[self.position]

        if ch == '"':
            self.position += 1
            start = self.position
            while self.position < len(self.source) and self.source[self.position] != '"':
                self.position += 1
            if self.position >= len(self.source):
                raise ValueError("String não fechada com aspas")
            string_val = self.source[start:self.position]
            self.position += 1
            self.next = Token("STRING", string_val)
            return

        if ch.isdigit():
            start = self.position
            self._advance_while(str.isdigit)
            num_str = self.source[start:self.position]
            self.next = Token("NUMBER", int(num_str))
            return

        if ch.isalpha() or ch == '_':
            start = self.position
            self._advance_while(lambda c: c.isalnum() or c == '_')
            ident = self.source[start:self.position]
            if ident in self.KEYWORDS:
                self.next = Token(self.KEYWORDS[ident], None)
            else:
                self.next = Token("IDENTIFIER", ident)
            return

        if ch == '=' and self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
            self.position += 2
            self.next = Token("EQUALS", None)
            return
        if ch == '&' and self.position + 1 < len(self.source) and self.source[self.position + 1] == '&':
            self.position += 2
            self.next = Token("AND", None)
            return
        if ch == '|' and self.position + 1 < len(self.source) and self.source[self.position + 1] == '|':
            self.position += 2
            self.next = Token("OR", None)
            return

        if ch in self.SIMPLE_TOKENS:
            self.position += 1
            self.next = Token(self.SIMPLE_TOKENS[ch], None)
            return

        if ch == '=':
            self.position += 1
            self.next = Token("ASSIGN", None)
            return
        if ch == '>':
            self.position += 1
            self.next = Token("GREATER", None)
            return
        if ch == '<':
            self.position += 1
            self.next = Token("LESS", None)
            return
        if ch == '!':
            self.position += 1
            self.next = Token("NOT", None)
            return

        raise ValueError(f"Caractere inválido: {ch}")

class SymbolEntry:
    def __init__(self, type_: str, value: Any = None, is_function: bool = False, ret_type: str = None):
        self.type = type_
        self.value = value
        self.is_function = is_function
        self.ret_type = ret_type

    def __repr__(self):
        return f"<SymbolEntry type={self.type} value={self.value} is_func={self.is_function}>"

class SymbolTable:
    def __init__(self, parent: Optional["SymbolTable"] = None):
        self.symbols: Dict[str, SymbolEntry] = {}
        self.parent = parent

    def declare(self, key: str, entry: SymbolEntry):
        if key in self.symbols:
            raise ValueError(f'Variável/funcção "{key}" já declarada neste escopo.')
        self.symbols[key] = entry

    def get_entry(self, key: str) -> SymbolEntry:
        if key in self.symbols:
            return self.symbols[key]
        if self.parent is not None:
            return self.parent.get_entry(key)
        raise ValueError(f'Variável não declarada: {key}')

    def get(self, key: str) -> Tuple[Any, str]:
        entry = self.get_entry(key)
        return entry.value, entry.type

    def set(self, key: str, value: Any, value_type: str):
        if key in self.symbols:
            entry = self.symbols[key]
        elif self.parent is not None:
            self.parent.set(key, value, value_type)
            return
        else:
            raise ValueError(f'Variável não declarada: {key}')

        if entry.is_function:
            raise ValueError(f'"{key}" é uma função, não pode receber atribuição')
        if entry.type != value_type:
            raise ValueError(f'Tentando atribuir {value_type} em variável do tipo {entry.type}')
        entry.value = value

class ReturnSignal(Exception):
    def __init__(self, value, value_type):
        super().__init__("return")
        self.value = value
        self.value_type = value_type

class Node:
    def evaluate(self, st: SymbolTable):
        raise NotImplementedError

class NoOp(Node):
    def evaluate(self, st):
        return None, None

class IntVal(Node):
    def __init__(self, value: int):
        self.value = value

    def evaluate(self, st):
        return self.value, "int"

class StrVal(Node):
    def __init__(self, value: str):
        self.value = value

    def evaluate(self, st):
        return self.value, "string"

class BoolVal(Node):
    def __init__(self, value: bool):
        self.value = value

    def evaluate(self, st):
        return self.value, "bool"

class Identifier(Node):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, st):
        return st.get(self.name)

    def generate(self, symbol_table, code):
        for statement in self.children:
            statement.generate(symbol_table, code)

class Println(Node):
    def __init__(self, expr: Node):
        self.expr = expr

    def evaluate(self, st):
        val, typ = self.expr.evaluate(st)
        if typ == "bool":
            print("true" if val else "false")
        else:
            print(val)
        return val, typ

class Scan(Node):
    def evaluate(self, st):
        user_in = input().strip()
        try:
            return int(user_in), "int"
        except ValueError:
            return user_in, "string"

class BinOp(Node):
    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, st):
        lv, lt = self.left.evaluate(st)
        rv, rt = self.right.evaluate(st)

        if self.op in ("==", ">", "<"):
            if lt != rt:
                raise ValueError("Não é possível comparar tipos diferentes")
            if self.op == "==":
                return lv == rv, "bool"
            if self.op == ">":
                if lt == "int":
                    return lv > rv, "bool"
                return lv > rv, "bool"
            if self.op == "<":
                if lt == "int":
                    return lv < rv, "bool"
                return lv < rv, "bool"

        if self.op in ("&&", "||"):
            if lt != "bool" or rt != "bool":
                raise ValueError("Operação lógica requer bool")
            return (lv and rv) if self.op == "&&" else (lv or rv), "bool"

        if self.op == "+":
            if lt == rt == "int":
                return lv + rv, "int"
            def to_str(v, t):
                if t == "bool":
                    return "true" if v else "false"
                return str(v)
            return to_str(lv, lt) + to_str(rv, rt), "string"
        if self.op == "-":
            if lt == rt == "int":
                return lv - rv, "int"
            raise ValueError("Operador - exige ints")
        if self.op == "*":
            if lt == rt == "int":
                return lv * rv, "int"
            raise ValueError("Operador * exige ints")
        if self.op == "/":
            if lt == rt == "int":
                if rv == 0:
                    raise ValueError("Divisão por zero")
                return lv // rv, "int"
            raise ValueError("Operador / exige ints")
        raise ValueError(f"Operador desconhecido {self.op}")

class UnOp(Node):
    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child

    def evaluate(self, st):
        v, t = self.child.evaluate(st)
        if self.op == '!':
            if t != 'bool':
                raise ValueError('! requer bool')
            return (not v), 'bool'
        if self.op == '+':
            if t != 'int':
                raise ValueError('+ unário requer int')
            return v, 'int'
        if self.op == '-':
            if t != 'int':
                raise ValueError('- unário requer int')
            return -v, 'int'
        raise ValueError(f'Operador unário desconhecido {self.op}')

class Assignment(Node):
    def __init__(self, identifier: Identifier, expr: Node):
        self.identifier = identifier
        self.expr = expr

    def evaluate(self, st):
        val, typ = self.expr.evaluate(st)
        st.set(self.identifier.name, val, typ)
        return val, typ

class VarDecl(Node):
    def __init__(self, name: str, var_type: str, expr: Optional[Node] = None):
        self.name = name
        self.var_type = var_type
        self.expr = expr

    def evaluate(self, st):
        st.declare(self.name, SymbolEntry(self.var_type, is_function=False))
        if self.expr is not None:
            val, typ = self.expr.evaluate(st)
            if typ != self.var_type:
                raise ValueError(
                    f"Incompatibilidade de tipo ao inicializar '{self.name}'. Esperado {self.var_type}, obteve {typ}"
                )
            st.set(self.name, val, typ)
            return val, typ
        return None, None

class Return(Node):
    def __init__(self, expr: Node):
        self.expr = expr

    def evaluate(self, st):
        val, typ = self.expr.evaluate(st)
        raise ReturnSignal(val, typ)

class Block(Node):
    def __init__(self, statements: List[Node]):
        self.statements = statements

    def evaluate(self, st):
        for stmt in self.statements:
            try:
                if isinstance(stmt, Block):
                    nested = SymbolTable(parent=st)
                    result = stmt.evaluate(nested)
                else:
                    result = stmt.evaluate(st)
            except ReturnSignal as rs:
                raise rs
        return None, None

class If(Node):
    def __init__(self, cond: Node, then_block: Block, else_block: Optional[Block]):
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block

    def evaluate(self, st):
        cond_val, cond_type = self.cond.evaluate(st)
        if cond_type != 'bool':
            raise ValueError('Condição do if deve ser bool')
        try:
            if cond_val:
                return self.then_block.evaluate(SymbolTable(parent=st))
            elif self.else_block is not None:
                return self.else_block.evaluate(SymbolTable(parent=st))
            return None, None
        except ReturnSignal as rs:
            raise rs

class For(Node):
    def __init__(self, cond: Node, block: Block):
        self.cond = cond
        self.block = block

    def evaluate(self, st):
        while True:
            cond_val, cond_type = self.cond.evaluate(st)
            if cond_type != 'bool':
                raise ValueError('Condição do for deve ser bool')
            if not cond_val:
                break
            try:
                self.block.evaluate(SymbolTable(parent=st))
            except ReturnSignal as rs:
                raise rs
        return None, None

class FuncDec(Node):
    def __init__(self, name: str, parameters: List[Tuple[str, str]], ret_type: str, body: Block):
        self.name = name
        self.parameters = parameters 
        self.ret_type = ret_type
        self.body = body

    def evaluate(self, st):
        st.declare(self.name, SymbolEntry("function", value=self, is_function=True, ret_type=self.ret_type))
        return None, None

class FuncCall(Node):
    def __init__(self, name: str, arg_exprs: List[Node]):
        self.name = name
        self.arg_exprs = arg_exprs

    def evaluate(self, st):
        entry = st.get_entry(self.name)
        if not entry.is_function:
            raise ValueError(f'"{self.name}" não é uma função')
        func_node: FuncDec = entry.value 
        if len(func_node.parameters) != len(self.arg_exprs):
            raise ValueError(f'Número incorreto de argumentos para {self.name}')

        call_table = SymbolTable(parent=st)
        for (param_name, param_type), arg_expr in zip(func_node.parameters, self.arg_exprs):
            val, typ = arg_expr.evaluate(st)
            if typ != param_type:
                raise ValueError(f'Tipo inválido para parâmetro {param_name}: esperado {param_type}, obteve {typ}')
            call_table.declare(param_name, SymbolEntry(param_type, value=val))

        try:
            func_node.body.evaluate(call_table)
        except ReturnSignal as rs:
            if func_node.ret_type != rs.value_type:
                raise ValueError(f'Função {self.name} deve retornar {func_node.ret_type}, obteve {rs.value_type}')
            return rs.value, rs.value_type
        if func_node.ret_type != 'void':
            raise ValueError(f'Função {self.name} deve retornar {func_node.ret_type}')
        return None, None
    
class Parser:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer

    def parse(self):
        program = self.parse_program()
        program.statements.append(FuncCall("main", []))
        if self.tok.next.type != 'EOF':
            raise ValueError('Tokens restantes após fim do programa')
        return program

    def consume(self, expected_type: str):
        if self.tok.next.type != expected_type:
            raise ValueError(f'Esperado {expected_type}, obteve {self.tok.next.type}')
        self.tok.select_next()

    def parse_program(self):
        stmts: List[Node] = []
        if self.tok.next.type == 'LBRACE':
            self.tok.select_next()
            while self.tok.next.type != 'RBRACE':
                stmts.append(self.parse_statement())
            self.consume('RBRACE')
        else:
            while self.tok.next.type != 'EOF':
                stmts.append(self.parse_statement())

        return Block(stmts)

    def parse_statement(self) -> Node:
        tok_type = self.tok.next.type

        if tok_type == 'VAR':
            self.tok.select_next()
            if self.tok.next.type != 'IDENTIFIER':
                raise ValueError('Esperado identificador após var')
            var_name = self.tok.next.value
            self.tok.select_next()
            if self.tok.next.type not in ('INT_TYPE', 'BOOL_TYPE', 'STRING_TYPE'):
                raise ValueError('Tipo inválido para variável')
            var_type_map = {
                'INT_TYPE': 'int',
                'BOOL_TYPE': 'bool',
                'STRING_TYPE': 'string',
            }
            var_type = var_type_map[self.tok.next.type]
            self.tok.select_next()
            init_expr = None
            if self.tok.next.type == 'ASSIGN':
                self.tok.select_next()
                init_expr = self.parse_b_expression()
            return VarDecl(var_name, var_type, init_expr)

        if tok_type == 'FUNC':
            return self.parse_func_declaration()

        if tok_type == 'RETURN':
            self.tok.select_next()
            expr = self.parse_b_expression()
            return Return(expr)

        if tok_type == 'PRINTLN':
            self.tok.select_next()
            self.consume('LPAREN')
            expr = self.parse_b_expression()
            self.consume('RPAREN')
            return Println(expr)

        if tok_type == 'LBRACE':
            self.tok.select_next()
            stmts: List[Node] = []
            while self.tok.next.type != 'RBRACE':
                stmts.append(self.parse_statement())
            self.consume('RBRACE')
            return Block(stmts)

        if tok_type == 'IF':
            self.tok.select_next()
            cond = self.parse_b_expression()
            self.consume('LBRACE')
            then_block = self.parse_block_contents()
            else_block = None
            if self.tok.next.type == 'ELSE':
                self.tok.select_next()
                self.consume('LBRACE')
                else_block = self.parse_block_contents()
            return If(cond, then_block, else_block)

        if tok_type == 'FOR':
            self.tok.select_next()
            cond = self.parse_b_expression()
            self.consume('LBRACE')
            block = self.parse_block_contents()
            return For(cond, block)

        if tok_type == 'IDENTIFIER':
            ident_name = self.tok.next.value
            self.tok.select_next()
            if self.tok.next.type == 'ASSIGN':
                self.tok.select_next()
                expr = self.parse_b_expression()
                return Assignment(Identifier(ident_name), expr)
            elif self.tok.next.type == 'LPAREN':
                args = self.parse_argument_list()
                return FuncCall(ident_name, args)
            else:
                raise ValueError('Esperado = ou ( após identificador')

        return self.parse_b_expression()

    def parse_block_contents(self) -> Block:
        stmts: List[Node] = []
        while self.tok.next.type != 'RBRACE':
            stmts.append(self.parse_statement())
        self.consume('RBRACE')
        return Block(stmts)

    def parse_func_declaration(self) -> FuncDec:
        self.tok.select_next()
        if self.tok.next.type != 'IDENTIFIER':
            raise ValueError('Esperado nome da função')
        func_name = self.tok.next.value
        self.tok.select_next()
        self.consume('LPAREN')
        params: List[Tuple[str, str]] = []
        if self.tok.next.type != 'RPAREN':
            while True:
                if self.tok.next.type != 'IDENTIFIER':
                    raise ValueError('Esperado nome do parâmetro')
                param_name = self.tok.next.value
                self.tok.select_next()
                if self.tok.next.type not in ('INT_TYPE', 'BOOL_TYPE', 'STRING_TYPE'):
                    raise ValueError('Tipo inválido de parâmetro')
                type_map = {
                    'INT_TYPE': 'int',
                    'BOOL_TYPE': 'bool',
                    'STRING_TYPE': 'string',
                }
                param_type = type_map[self.tok.next.type]
                self.tok.select_next()
                params.append((param_name, param_type))
                if self.tok.next.type == 'COMMA':
                    self.tok.select_next()
                    continue
                break
        self.consume('RPAREN')
        self.tok._advance_while(str.isspace)
        
        if self.tok.next.type not in ('INT_TYPE', 'BOOL_TYPE', 'STRING_TYPE'):
            current_pos = self.tok.position
            while current_pos < len(self.tok.source) and self.tok.source[current_pos].isspace():
                current_pos += 1
            
            if current_pos < len(self.tok.source):
                if current_pos + 3 <= len(self.tok.source) and self.tok.source[current_pos:current_pos+3] == "int":
                    ret_type = "int"
                    self.tok.position = current_pos + 3
                    self.tok.select_next()
                elif current_pos + 4 <= len(self.tok.source) and self.tok.source[current_pos:current_pos+4] == "bool":
                    ret_type = "bool"
                    self.tok.position = current_pos + 4
                    self.tok.select_next()
                elif current_pos + 6 <= len(self.tok.source) and self.tok.source[current_pos:current_pos+6] == "string":
                    ret_type = "string"
                    self.tok.position = current_pos + 6
                    self.tok.select_next()
                else:
                    ret_type = "void"
            else:
                ret_type = "void"
        else:
            ret_type = {
                'INT_TYPE': 'int',
                'BOOL_TYPE': 'bool',
                'STRING_TYPE': 'string',
                'VOID_TYPE': 'void',
            }[self.tok.next.type]
            self.tok.select_next()
        
        self.consume('LBRACE')
        body_block = self.parse_block_contents()
        return FuncDec(func_name, params, ret_type, body_block)

    def parse_argument_list(self) -> List[Node]:
        self.consume('LPAREN')
        args: List[Node] = []
        if self.tok.next.type != 'RPAREN':
            while True:
                args.append(self.parse_b_expression())
                if self.tok.next.type == 'COMMA':
                    self.tok.select_next()
                    continue
                break
        self.consume('RPAREN')
        return args

    def parse_b_expression(self):
        node = self.parse_b_term()
        while self.tok.next.type == 'OR':
            self.tok.select_next()
            node = BinOp('||', node, self.parse_b_term())
        return node

    def parse_b_term(self):
        node = self.parse_rel_expression()
        while self.tok.next.type == 'AND':
            self.tok.select_next()
            node = BinOp('&&', node, self.parse_rel_expression())
        return node

    def parse_rel_expression(self):
        node = self.parse_expression()
        while self.tok.next.type in ('LESS', 'GREATER', 'EQUALS'):
            if self.tok.next.type == 'LESS':
                self.tok.select_next()
                node = BinOp('<', node, self.parse_expression())
            elif self.tok.next.type == 'GREATER':
                self.tok.select_next()
                node = BinOp('>', node, self.parse_expression())
            elif self.tok.next.type == 'EQUALS':
                self.tok.select_next()
                node = BinOp('==', node, self.parse_expression())
        return node

    def parse_expression(self):
        node = self.parse_term()
        while self.tok.next.type in ('PLUS', 'MINUS'):
            if self.tok.next.type == 'PLUS':
                self.tok.select_next()
                node = BinOp('+', node, self.parse_term())
            else:
                self.tok.select_next()
                node = BinOp('-', node, self.parse_term())
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.tok.next.type in ('MULTIPLY', 'DIVIDE'):
            if self.tok.next.type == 'MULTIPLY':
                self.tok.select_next()
                node = BinOp('*', node, self.parse_factor())
            else:
                self.tok.select_next()
                node = BinOp('/', node, self.parse_factor())
        return node

    def parse_factor(self):
        tok_type = self.tok.next.type
        if tok_type == 'NUMBER':
            val = IntVal(self.tok.next.value)
            self.tok.select_next()
            return val
        if tok_type == 'STRING':
            val = StrVal(self.tok.next.value)
            self.tok.select_next()
            return val
        if tok_type == 'IDENTIFIER':
            ident_name = self.tok.next.value
            self.tok.select_next()
            if self.tok.next.type == 'LPAREN':
                args = self.parse_argument_list()
                return FuncCall(ident_name, args)
            return Identifier(ident_name)
        if tok_type == 'LPAREN':
            self.tok.select_next()
            expr = self.parse_b_expression()
            self.consume('RPAREN')
            return expr
        if tok_type == 'PLUS':
            self.tok.select_next()
            return UnOp('+', self.parse_factor())
        if tok_type == 'MINUS':
            self.tok.select_next()
            return UnOp('-', self.parse_factor())
        if tok_type == 'NOT':
            self.tok.select_next()
            return UnOp('!', self.parse_factor())
        if tok_type == 'SCAN':
            self.tok.select_next()
            self.consume('LPAREN')
            self.consume('RPAREN')
            return Scan()
        raise ValueError(f'Fator inválido ou não suportado: {tok_type}')

def run(code: str):
    filtered = PrePro.filter(code)
    tk = Tokenizer(filtered)
    parser = Parser(tk)
    ast_root = parser.parse()
    global_st = SymbolTable() 
    try:
        ast_root.evaluate(global_st)
    except ReturnSignal:
        raise ValueError('Return fora de função')

def main():
    if len(sys.argv) != 2:
        print("Uso: python3 main.py <arquivo>")
        sys.exit(1)
    file_path = sys.argv[1]
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        run(code)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
    except Exception as e:
        print(f"Erro: {e}")
        raise e

if __name__ == '__main__':
    main()