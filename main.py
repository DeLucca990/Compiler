import sys

if len(sys.argv) != 2:
    print("Deve ser passado um argumento para o programa")
    raise ValueError("Número de argumentos inválido")

exp = sys.argv[1]

result = 0
current_number = ""
current_operator = "+"

if exp == '':
    print("Expressão vazia")
    raise ValueError("Expressão vazia")
if exp[-1] in "+-":
    print("Último caractere não pode ser um operador")
    raise ValueError("Último caractere não pode ser um operador")
if "+" not in exp or "-" not in exp:
    print("Expressão inválida")
    raise ValueError("Expressão inválida")

for i in range(len(exp)):
    if exp[i].isdigit():
        current_number += exp[i]
    elif exp[i] in "+-":
        if current_operator == "+":
            result += int(current_number)
        elif current_operator == "-":
            result -= int(current_number)
        current_operator = exp[i]
        current_number = ""

if current_number:
    if current_operator == "+":
        result += int(current_number)
    elif current_operator == "-":
        result -= int(current_number)

print(result)