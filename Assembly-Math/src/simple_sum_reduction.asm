  section .text 
; avx512 
global sum_horizontal_avx512
global sum_vertical_avx512
; avx2
global sum_horizontal_avx2
global sum_vertical_avx2
; avx 
global sum_horizontal_avx 
global sum_vertical_avx

; extern "C" float sum_type_avxXXX(float* x);
; x ptr to float is in rdi 

; rsi holds reference to size but we don't use it 
sum_horizontal_avx512:
  mov r11, [rdi]    ; Dereference array pointer
  mov r10, [rsi]
  vmovaps zmm0, [r11]

  ; Pull upper 8 floats from 512-bit vector to 256-bit vector 
  vextractf32x8 ymm1, zmm0, 1
  vaddps ymm0, ymm0, ymm1       ; Packed Add 

  ; Place upper section of 256-bit register in 128-bit 
  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1

  ; Horizontal addition in 128-bit register to reduce sum
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  ; Return sum in xmm0
  ret

sum_vertical_avx512:
  ; Functionally identical for the first section 
  mov r11, [rdi]    ; Dereference array pointer
  mov r10, [rsi]
  vmovaps zmm0, [r11]
  vextractf32x8 ymm1, zmm0, 1
  vaddps ymm0, ymm0, ymm1 
  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1

  ; Vertical addition using shuffle
  ; Reduce to 2 floats from 4 
  vpermilps xmm1, xmm0, 0b10110001
  vaddps xmm0, xmm0, xmm1

  ; Second shuffle to reduce to one float 
  vpermilps xmm1, xmm0, 0b01001110
  vaddps xmm0, xmm0, xmm1
  
  ret

sum_horizontal_avx2:
  mov r11, [rdi]    ; Dereference array pointer
  mov r10, [rsi]
  vmovaps ymm0, [r11]        ; load lower 8 floats to ymm0 
  vmovaps ymm1, [r11 + 32]   ; upper 8 floats into ymm1

  vaddps ymm0, ymm0, ymm1    ; add upper 8 to lower 8 floats

  vextractf128 xmm1, ymm0, 1 ; upper 4 floats into xmm1  
  vaddps xmm0, xmm0, xmm1    ; add upper 4 to lower 4

  vhaddps xmm0, xmm0, xmm0   ; horizontal add 
  vhaddps xmm0, xmm0, xmm0   ; final reduction 

  ret 

sum_vertical_avx2: 
  mov r11, [rdi]    ; Dereference array pointer
  mov r10, [rsi]
  ; Identical reduction to horizontal avx2
  vmovaps ymm0, [r11]
  vmovaps ymm1, [r11 + 32]

  vextractf128 xmm1, ymm0, 1 
  vaddps xmm0, xmm0, xmm1

  ; Identical shuffle and vertical add to avx512
  vpermilps xmm1, xmm0, 0b10110001 
  vaddps xmm0, xmm0, xmm1 

  vpermilps xmm1, xmm0, 0b01001110 
  vaddps xmm0, xmm0, xmm1 

  ret 

sum_horizontal_avx:
  mov r11, [rdi]    ; Dereference array pointer
  mov r10, [rsi]
  ; Filter rdi into 128-bit registers
  vmovaps xmm0, [r11]         ; first 4 floats 
  vmovaps xmm1, [r11 + 16]    ; next 4 
  vmovaps xmm2, [r11 + 32]    ; etc.
  vmovaps xmm3, [r11 + 48]

  ; Reduce registers to single 128-bit register
  vaddps xmm0, xmm0, xmm1 
  vaddps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2 

  ; Reduce through horizontal packed addition 
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  ret 

sum_vertical_avx:
  mov r11, [rdi]    ; Dereference array pointer
  mov r10, [rsi]
  ; Identical Filter and reduction to horizontal avx
  vmovaps xmm0, [r11]          
  vmovaps xmm1, [r11 + 16]     
  vmovaps xmm2, [r11 + 32]    
  vmovaps xmm3, [r11 + 48]

  vaddps xmm0, xmm0, xmm1 
  vaddps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2 

  ; Standard vertical shuffle and packed add 
  vpermilps xmm1, xmm0, 0b1011001 
  vaddps xmm0, xmm0, xmm1 
  vpermilps xmm1, xmm0, 0b1001110 
  vaddps xmm0, xmm0, xmm1 

  ret 


