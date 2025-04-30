section .text
_start:
    sub esp, 4    ; aloca i
    sub esp, 4    ; aloca n
    sub esp, 4    ; aloca f
    mov eax, 1
    mov [ebp-12], eax    ; inicializa f
    ; call scanf
    mov [ebp-8], eax    ; atribui n
    mov eax, 2
    mov [ebp-4], eax    ; atribui i
    start_28:
    mov eax,[ebp-4]
    push eax    ; salva L
    mov eax,[ebp-8]
    push eax    ; salva L
    mov eax, 1
    mov ecx, eax    ; R em ECX
    pop eax        ; L em EAX
    add eax, ecx
    mov ecx, eax    ; R em ECX
    pop eax        ; L em EAX
    cmp eax, ecx
    setl al
    movzx eax, al
    cmp eax,0
    je end_28
    mov eax,[ebp-12]
    push eax    ; salva L
    mov eax,[ebp-4]
    mov ecx, eax    ; R em ECX
    pop eax        ; L em EAX
    imul eax, ecx
    mov [ebp-12], eax    ; atribui f
    mov eax,[ebp-4]
    push eax    ; salva L
    mov eax, 1
    mov ecx, eax    ; R em ECX
    pop eax        ; L em EAX
    add eax, ecx
    mov [ebp-4], eax    ; atribui i
    jmp start_28
    end_28:
    mov eax,[ebp-12]
    push eax
    call print_int
    add esp, 4
    mov eax, 1
    mov ebx, 0
    int 0x80
