import sys

if len(sys.argv) != 2:
    print("Deve ser passado um argumento para o programa")
    sys.exit(1)

exp = sys.argv[1]

result = 0
current_number = ""
current_operator = "+"

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