"""NCSL Virtual Machine — Direct bytecode execution."""
import struct
from .token_table import *

class NCSLVM:
    def __init__(self):
        self.code = b""; self.ip = 0; self.stack = []; self.vars = {}
    
    def load(self, binary: bytes):
        if binary[:2] != MAGIC: raise ValueError("Bad magic")
        self.code = binary; self.ip = 8; self.stack = []; self.vars = {}
    
    def run(self):
        while self.ip < len(self.code):
            t = self.code[self.ip]; self.ip += 1
            if t == NOP: pass
            elif t == END: return self.stack[-1] if self.stack else None
            elif t == RETURN: return self.stack.pop() if self.stack else None
            elif t == I8: self.stack.append(struct.unpack_from('<b', self.code, self.ip)[0]); self.ip += 1
            elif t == I16: self.stack.append(struct.unpack_from('<h', self.code, self.ip)[0]); self.ip += 2
            elif t == I32: self.stack.append(struct.unpack_from('<i', self.code, self.ip)[0]); self.ip += 4
            elif t == F64: self.stack.append(struct.unpack_from('<d', self.code, self.ip)[0]); self.ip += 8
            elif t == BOOL: self.stack.append(self.code[self.ip] != 0); self.ip += 1
            elif t == STR:
                length = self.code[self.ip]; self.ip += 1
                if length == 0: length = struct.unpack_from('<H', self.code, self.ip)[0]; self.ip += 2
                self.stack.append(self.code[self.ip:self.ip+length].decode('utf-8')); self.ip += length
            elif t == NONE: self.stack.append(None)
            elif t == LIST:
                count = struct.unpack_from('<H', self.code, self.ip)[0]; self.ip += 2
                self.stack.append([self._read() for _ in range(count)])
            elif t == DICT:
                count = struct.unpack_from('<H', self.code, self.ip)[0]; self.ip += 2
                self.stack.append({self._read(): self._read() for _ in range(count)})
            elif t == ADD: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a + b)
            elif t == SUB: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a - b)
            elif t == MUL: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a * b)
            elif t == DIV: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a / b if b else 0)
            elif t == EQ: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a == b)
            elif t == LT: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a < b)
            elif t == GT: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a > b)
            elif t == AND: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a and b)
            elif t == OR: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a or b)
            elif t == NOT: self.stack.append(not self.stack.pop())
            elif t == IF:
                if not self.stack.pop(): self._skip()
            elif t == ELSE: self._skip_end()
            elif t == WHILE: pass
            elif t == FOR:
                var = self.stack.pop(); items = self.stack.pop()
                for item in items: self.vars[var] = item
                self._skip()
            elif t == CALL:
                name = self.stack.pop(); args = []
                while self.ip < len(self.code) and self.code[self.ip] != END:
                    args.insert(0, self.stack.pop())
                self.ip += 1
                if name == "print": print(*args); self.stack.append(None)
                elif name == "len": self.stack.append(len(args[0]) if args else 0)
                elif name == "range": self.stack.append(list(range(*args)))
                elif name in self.vars: self.stack.append(self.vars[name])
            elif t == PRINT: print(self.stack.pop())
            elif t == RAISE:
                e = self.stack.pop() if self.stack else "Error"
                raise Exception(str(e))
            elif t == BREAK: self._skip()
            elif t == CONTINUE: self._skip()
        return self.stack[-1] if self.stack else None
    
    def _read(self):
        t = self.code[self.ip]; self.ip += 1
        if t == I8: v = struct.unpack_from('<b', self.code, self.ip)[0]; self.ip += 1; return v
        elif t == I16: v = struct.unpack_from('<h', self.code, self.ip)[0]; self.ip += 2; return v
        elif t == I32: v = struct.unpack_from('<i', self.code, self.ip)[0]; self.ip += 4; return v
        elif t == STR:
            length = self.code[self.ip]; self.ip += 1
            if length == 0: length = struct.unpack_from('<H', self.code, self.ip)[0]; self.ip += 2
            v = self.code[self.ip:self.ip+length].decode('utf-8'); self.ip += length; return v
        elif t == NONE: return None
        elif t == BOOL: v = self.code[self.ip] != 0; self.ip += 1; return v
        return None
    
    def _skip(self):
        d = 0
        while self.ip < len(self.code):
            if self.code[self.ip] in (IF, WHILE, FOR, TRY): d += 1
            elif self.code[self.ip] == END:
                if d == 0: self.ip += 1; return
                d -= 1
            self.ip += 1
    
    def _skip_end(self):
        d = 0
        while self.ip < len(self.code):
            if self.code[self.ip] in (IF, WHILE, FOR): d += 1
            elif self.code[self.ip] == END:
                if d == 0: return
                d -= 1
            self.ip += 1
