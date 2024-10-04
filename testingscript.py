# Code Converted from Assembly to Python

# Initialize registers
ax = 0
ds = ax
es = ax
ss = ax
sp = 0x7c00

# Test Disk Extension
DriveId = dl
ah = 0x41
bx = 0x55aa
int 0x13
if carry_flag:
    goto NotSupport
if bx != 0xaa55:
    goto NotSupport

# Load Loader
si = ReadPacket
word[si] = 0x10
word[si+2] = 5
word[si+4] = 0x7e00
word[si+6] = 0
dword[si+8] = 1
dword[si+0xc] = 0
dl = DriveId
ah = 0x42
int 0x13
if carry_flag:
    goto ReadError

dl = DriveId
jmp 0x7e00

ReadError:
NotSupport:
ah = 0x13
al = 1
bx = 0xa
dx = 0
bp = Message
cx = MessageLen
int 0x10

End:
hlt
jmp End

DriveId = 0
Message = "We have an error in boot process"
MessageLen = len(Message)
ReadPacket = [0] * 16

[0x1be-($-$$)] = [0] * (0x1be-($-$$))

db 0x80
db 0, 2, 0
db 0xf0
db 0xff, 0xff, 0xff
dd 1
dd (20*16*63-1)

[16*3] = [0] * (16*3)

db 0x55
db 0xaa
