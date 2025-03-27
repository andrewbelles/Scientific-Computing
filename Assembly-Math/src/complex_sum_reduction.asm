  section .text 
; Quicksums for all avx types

global quicksum_havx512 
global quicksum_vavx512

global quicksum_havx2 
global quicksum_vavx2

global quicksum_havx 
global quicksum_vavx

; extern "C" float quicksum_Xavx000(float* x, size_t n)
; x is an array of real values
; n is the multiple of 16 that x is sized to  

quicksum_havx512:
  ; Dereference by one 
  mov r11, [rdi]
  mov r10, [rsi]

  ; Get multiple of 16 
  shr r10, 4

  ; Init counter 
  xor rcx, rcx
  vxorps xmm2, xmm2 
  
.L1:

  ; Identical to simple approach but rcx * 512 index for 16 float batch 
  mov r8, rcx 
  shl r8, 6                   ; Ptr index is multiplied by 64
  vmovaps zmm0, [r11 + r8]    ; Next 16 floats

  vextractf32x8 ymm1, zmm0, 1 
  vaddps ymm0, ymm0, ymm1 

  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1 

  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0 

  vaddps xmm2, xmm2, xmm0 

  inc rcx 
  cmp rcx, r10 
  jl .L1                             ; Loop while lt 

  ; Shift saved sum back to xmm0 to return 
  vmovaps xmm0, xmm2
  ret

quicksum_vavx512:
  ; Dereference by one 
  mov r11, [rdi]
  mov r10, [rsi]

  ; Get multiple of 16 
  shr r10, 4

  xor rcx, rcx
  vxorps xmm2, xmm2 
  
.L1:
  ; Identical to horizontal to load and register reduction
  mov r8, rcx 
  shl r8, 6                   
  vmovaps zmm0, [r11 + r8]    

  vextractf32x8 ymm1, zmm0, 1 
  vaddps ymm0, ymm0, ymm1 

  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1 

  vpermilps xmm1, xmm0, 0b10110001
  vaddps xmm0, xmm0, xmm1

  vpermilps xmm1, xmm0, 0b01001110
  vaddps xmm0, xmm0, xmm1

  vaddps xmm2, xmm2, xmm0 

  inc rcx 
  cmp rcx, r10 
  jl .L1                             ; Loop while lt 

  ; Shift saved sum back to xmm0 to return 
  vmovaps xmm0, xmm2
  ret

quicksum_havx2:
  ; Dereference by one 
  mov r11, [rdi]
  mov r10, [rsi]

  ; Get multiple of 16 
  shr r10, 4

  xor rcx, rcx
  vxorps xmm2, xmm2 
  
.L1:

  ; Identical but we want to load lower 32 bits into one register and upper to other
  mov r8, rcx 
  shl r8, 6                         ; Ptr index is multiplied by 32
  vmovaps ymm0, [r11 + r8]          ; Lower 8 floats
  vmovaps ymm1, [r11 + r8 + 32]          ; Upper 8 floats

  vaddps ymm0, ymm0, ymm1 

  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1 

  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0 

  vaddps xmm2, xmm2, xmm0 

  inc rcx 
  cmp rcx, r10 
  jl .L1                            ; Loop while lt 

  ; Shift saved sum back to xmm0 to return 
  vmovaps xmm0, xmm2
  ret

quicksum_vavx2:
  ; Dereference by one 
  mov r11, [rdi]
  mov r10, [rsi]

  ; Get multiple of 16 
  shr r10, 4

  xor rcx, rcx
  vxorps xmm2, xmm2 
  
.L1:
  ; Identical to horizontal for multi-register load 
  mov r8, rcx 
  shl r8, 6                         ; Ptr index is multiplied by 32
  vmovaps ymm0, [r11 + r8]          ; Lower 8 floats
  vmovaps ymm1, [r11 + r8 + 32]     ; Upper 8 floats

  vaddps ymm0, ymm0, ymm1 

  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1 

  ; Vertical shuffle and add
  vpermilps xmm1, xmm0, 0b10110001
  vaddps xmm0, xmm0, xmm1

  vpermilps xmm1, xmm0, 0b01001110
  vaddps xmm0, xmm0, xmm1

  vaddps xmm2, xmm2, xmm0 

  inc rcx 
  cmp rcx, r10 
  jl .L1                             ; Loop while lt 

  ; Shift saved sum back to xmm0 to return 
  vmovaps xmm0, xmm2
  ret

quicksum_havx:
  ; Dereference by one 
  mov r11, [rdi]
  mov r10, [rsi]

  ; Get multiple of 16 
  shr r10, 4

  xor rcx, rcx
  vxorps xmm4, xmm4 
  
.L1:
  ; Load 16 floats into 4 128-bit registers 
  mov r8, rcx 
  shl r8, 6                  ; Multiply index by 16  
  vmovaps xmm0, [r11 + r8]  
  vmovaps xmm1, [r11 + r8 + 16]    
  vmovaps xmm2, [r11 + r8 + 32]
  vmovaps xmm3, [r11 + r8 + 48]

  vaddps xmm0, xmm0, xmm1 
  vaddps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2

  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1 

  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0 

  vaddps xmm4, xmm4, xmm0 

  inc rcx 
  cmp rcx, r10
  jl .L1                             ; Loop while lt 

  ; Shift saved sum back to xmm0 to return 
  vmovaps xmm0, xmm4
  ret

quicksum_vavx:
  ; Dereference by one 
  mov r11, [rdi]
  mov r10, [rsi]

  ; Get multiple of 16 
  shr r10, 4

  xor rcx, rcx
  vxorps xmm4, xmm4 
  
.L1:
  ; Identical to horizontal algorithm
  mov r8, rcx 
  shl r8, 6                  ; Multiply index by 16  
  vmovaps xmm0, [r11 + r8]  
  vmovaps xmm1, [r11 + r8 + 16]    
  vmovaps xmm2, [r11 + r8 + 32]
  vmovaps xmm3, [r11 + r8 + 48]

  vaddps xmm0, xmm0, xmm1 
  vaddps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2

  vpermilps xmm1, xmm0, 0b10110001
  vaddps xmm0, xmm0, xmm1

  vpermilps xmm1, xmm0, 0b01001110
  vaddps xmm0, xmm0, xmm1

  vaddps xmm4, xmm4, xmm0 

  inc rcx 
  cmp rcx, r10
  jl .L1                             ; Loop while lt 

  ; Shift saved sum back to xmm0 to return 
  vmovaps xmm0, xmm4
  ret
