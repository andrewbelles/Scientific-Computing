section .text
global merge_sort
extern malloc, free

; extern "C" int merge_sort(float*& x, size_t& n);
; rdi holds pointer to x 
; rsi is true size 

merge_sort:
  ; prologue
  push rbp
  mov rbp, rsp 
  push rbx
  push r12 
  push r13 
  push r14 
  push r15
  
  ; Clear Comparison Count
  xor r15, r15
  ; Ensure 16-byte stack alignment
  sub rsp, 8
  ; Save ptr to array and size
  ; Super important that [] is used to dereference as they are pbr
  mov r12, [rdi]        ; Store array pointer in r12
  mov r13, [rsi]        ; Store size in r13
  ; Create temporary array 
  mov rdi, r13        ; Size in elements
  shl rdi, 2          
  call malloc wrt ..plt 
  mov r14, rax        ; temp array to r14  

  ; Initialize width to 1
  mov r9, 1
.L_outer:
  ; Check if width >= array size
  cmp r9, r13
  jge .sort_done
  
  ; Initialize i to 0
  xor r10, r10
.L_inner:
  ; Calculate left, mid, right
  mov rcx, r10        ; left = i
  
  ; Calculate mid = min(left + width, n)
  mov rax, rcx
  add rax, r9
  cmp rax, r13
  cmovae rax, r13
  
  ; Calculate right = min(left + 2*width, n)
  mov rdx, r9
  shl rdx, 1
  add rdx, rcx
  cmp rdx, r13
  cmovae rdx, r13
  
  ; Skip merge if left == mid or mid == right
  cmp rcx, rax
  je .skip_merge
  cmp rax, rdx
  je .skip_merge
  
  ; Call merge function
  push r9
  push r10
  push r13
  push r14
  
  mov rdi, r12        ; array
  mov rsi, r14        ; temp
  ; rcx, rax, rdx already set to left, mid, right
  call merge_subarrays
  
  pop r14
  pop r13
  pop r10
  pop r9
  
.skip_merge:
  ; Increment i by 2*width
  add r10, r9
  add r10, r9
  
  ; Continue inner loop if i < n
  cmp r10, r13
  jl .L_inner
  
  ; Double width and continue outer loop
  shl r9, 1
  jmp .L_outer
  
.sort_done:
  ; Free temporary array
  mov rdi, r14
  call free wrt ..plt
  
.sort_cleanup:
  ; Restore stack alignment
  add rsp, 8
  
  mov rax, r15
  
  ; epilogue
  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  ret

; merge_subarrays(float* arr, float* temp, size_t left, size_t mid, size_t right)
; Parameters:
;   rdi = array pointer
;   rsi = temp array pointer
;   rcx = left index
;   rax = mid index
;   rdx = right index
merge_subarrays:
  push rbp
  mov rbp, rsp
  push rbx
  push r12
  push r13
  push r14
  
  ; Save parameters
  mov r12, rcx        ; r12 = left
  mov r13, rax        ; r13 = mid
  mov r14, rdx        ; r14 = right
  mov rbx, rcx        ; rbx = k (index into temp)
  
  ; i = left, j = mid
  mov rcx, r12        ; i = left
  mov rdx, r13        ; j = mid
  
  ; Main merge loop
.L1:
  ; Check if either subarray is exhausted
  cmp rcx, r13        ; i < mid?
  jge .L2
  
  cmp rdx, r14        ; j < right?
  jge .L3
  
  ; Compare arr[i] and arr[j]
  vmovss xmm0, [rdi + 4*rcx]
  vmovss xmm1, [rdi + 4*rdx]
  vcomiss xmm0, xmm1
  inc r15 
  
  ja .j_value
  
  ; Take from left
  vmovss [rsi + 4*rbx], xmm0
  inc rcx
  jmp .pass
  
.j_value:
  vmovss [rsi + 4*rbx], xmm1
  inc rdx
  
.pass:
  inc rbx
  jmp .L1
  
.L2:
  ; Copy remaining elements from right subarray
  cmp rdx, r14
  jge .copy_back
  
  vmovss xmm0, [rdi + 4*rdx]
  vmovss [rsi + 4*rbx], xmm0
  inc rdx
  inc rbx
  jmp .L2
  
.L3:
  ; Copy remaining elements from left subarray
  cmp rcx, r13
  jge .copy_back
  
  vmovss xmm0, [rdi + 4*rcx]
  vmovss [rsi + 4*rbx], xmm0
  inc rcx
  inc rbx
  jmp .L3
  
.copy_back:
  ; Copy back from temp to original array
  mov rcx, r12        ; rcx = left
  
.L4:
  cmp rcx, r14        ; rcx < right
  jge .merge_cleanup
  
  vmovss xmm0, [rsi + 4*rcx]
  vmovss [rdi + 4*rcx], xmm0
  inc rcx
  jmp .L4
  
.merge_cleanup:
  ; Epilogue
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  ret
