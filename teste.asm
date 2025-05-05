section .data
    format_out: db "%d", 10, 0 ; format do printf
    format_in : db "%d", 0 ; format do scanf
    scan_int  : dd 0 ; 32-bits integer

section .text
    extern printf ; usar _printf para Windows
    extern scanf ; usar _scanf para Windows
    extern _ExitProcess@4 ; usar para Windows
    global _start ; início do programa
_start:
    push ebp            ; guarda o EBP
    mov ebp, esp       ; zera a pilha
    sub esp, 4    ; aloca i
    sub esp, 4    ; aloca n
    sub esp, 4    ; aloca f
    mov eax, 1
    mov [ebp-12], eax    ; inicializa f
    push scan_int
    push format_in
    call scanf
    add esp, 8
    mov eax, dword [scan_int]
    mov [ebp-8], eax    ; atribui n
    mov eax, 2
    mov [ebp-4], eax    ; atribui i
    loop_28:
    mov eax,[ebp-4]
    push eax
    mov eax,[ebp-8]
    push eax
    mov eax, 1
    mov ecx, eax
    pop eax
    add eax, ecx
    mov ecx, eax
    pop eax
    cmp eax, ecx
    setl al
    movzx eax, al
    cmp eax,0
    je end_28
    mov eax,[ebp-12]
    push eax
    mov eax,[ebp-4]
    mov ecx, eax
    pop eax
    imul eax, ecx
    mov [ebp-12], eax    ; atribui f
    mov eax,[ebp-4]
    push eax
    mov eax, 1
    mov ecx, eax
    pop eax
    add eax, ecx
    mov [ebp-4], eax    ; atribui i
    jmp loop_28
    end_28:
    mov eax,[ebp-12]
    push eax
    add esp, 4
    push eax
    push format_out
    call printf
    add esp, 8
    mov eax, 1
    mov ebx, 0
    int 0x80
    mov esp, ebp       ; reestabelece a pilha
    pop ebp
    ; chamada da interrupcao de saida (Linux)
    mov eax, 1
    xor ebx, ebx
    int 0x80
    ; Para Windows:
    ; push dword 0
    ; call _ExitProcess@4
