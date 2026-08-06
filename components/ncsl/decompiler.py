"""NCSL Decompiler — NCSL binary → Python source."""
import struct
from .token_table import *

class NCSLDecompiler:
    def __init__(self):
        self.code = b""
        self.pos = 0
        self.out = []
        self.indent = 0
    
    def decompile(self, binary: bytes) -> str:
        if binary[:2] != MAGIC:
            raise ValueError("Bad magic")
        self.code = binary
        self.pos = 8
        self.out = []
        self.indent = 0
        self._body()
        return '\n'.join(self.out)
    
    def _emit(self, line):
        self.out.append('    ' * self.indent + line)
    
    def _peek(self):
        return self.code[self.pos] if self.pos < len(self.code) else None
    
    def _body(self):
        while self.pos < len(self.code):
            t = self.code[self.pos]
            
            if t == END:
                self.pos += 1
                return
            elif t == NOP:
                self.pos += 1
                self._emit('pass')
            elif t == RETURN:
                self.pos += 1
                self._emit(f'return {self._read()}')
            elif t == FUNC_DEF:
                self.pos += 1
                name = self._read()
                args = []
                while self._peek() != END:
                    args.append(self._read())
                self.pos += 1  # Skip END after args
                self._emit(f'def {name}({", ".join(args)}):')
                self.indent += 1
                self._body()  # Function body
                self.indent -= 1
            elif t == CALL:
                self.pos += 1
                name = self._read()
                args = []
                while self.pos < len(self.code) and self.code[self.pos] != END:
                    args.append(self._read())
                if self.pos < len(self.code):
                    self.pos += 1  # Skip END
                self._emit(f'{name}({", ".join(str(a) for a in args)})')
            elif t == IF:
                self.pos += 1
                cond = self._read()
                self._emit(f'if {cond}:')
                self.indent += 1
                self._body()
                self.indent -= 1
                if self._peek() == ELSE:
                    self.pos += 1
                    self._emit('else:')
                    self.indent += 1
                    self._body()
                    self.indent -= 1
            elif t == FOR:
                self.pos += 1
                var = self._read()
                it = self._read()
                self._emit(f'for {var} in {it}:')
                self.indent += 1
                self._body()
                self.indent -= 1
            elif t == WHILE:
                self.pos += 1
                cond = self._read()
                self._emit(f'while {cond}:')
                self.indent += 1
                self._body()
                self.indent -= 1
            elif t == CLASS_DEF:
                self.pos += 1
                name = self._read()
                self._emit(f'class {name}:')
                self.indent += 1
                self._body()
                self.indent -= 1
            elif t == METHOD_DEF:
                self.pos += 1
                name = self._read()
                args = []
                while self._peek() != END:
                    args.append(self._read())
                self.pos += 1
                self._emit(f'def {name}({", ".join(args)}):')
                self.indent += 1
                self._body()
                self.indent -= 1
            elif t == PRINT:
                self.pos += 1
                self._emit(f'print({self._read()})')
            elif t == TRY:
                self.pos += 1
                self._emit('try:')
                self.indent += 1
                self._body()
                self.indent -= 1
                while self._peek() == EXCEPT:
                    self.pos += 1
                    self._emit('except:')
                    self.indent += 1
                    self._body()
                    self.indent -= 1
            elif t == RAISE:
                self.pos += 1
                self._emit(f'raise {self._read()}')
            elif t == ASSERT:
                self.pos += 1
                self._emit(f'assert {self._read()}')
            elif t == BREAK:
                self.pos += 1
                self._emit('break')
            elif t == CONTINUE:
                self.pos += 1
                self._emit('continue')
            else:
                # Unknown token — skip it
                val = self._read()
                self._emit(f'# {val}')
    
    def _read(self):
        t = self.code[self.pos]
        self.pos += 1
        
        if t == STR:
            length = self.code[self.pos]
            self.pos += 1
            if length == 0:
                length = struct.unpack_from('<H', self.code, self.pos)[0]
                self.pos += 2
            s = self.code[self.pos:self.pos+length].decode('utf-8')
            self.pos += length
            return repr(s)
        elif t == I8:
            v = struct.unpack_from('<b', self.code, self.pos)[0]
            self.pos += 1
            return v
        elif t == I16:
            v = struct.unpack_from('<h', self.code, self.pos)[0]
            self.pos += 2
            return v
        elif t == I32:
            v = struct.unpack_from('<i', self.code, self.pos)[0]
            self.pos += 4
            return v
        elif t == F64:
            v = struct.unpack_from('<d', self.code, self.pos)[0]
            self.pos += 8
            return v
        elif t == BOOL:
            v = self.code[self.pos] != 0
            self.pos += 1
            return v
        elif t == NONE:
            return 'None'
        elif t == LIST:
            count = struct.unpack_from('<H', self.code, self.pos)[0]
            self.pos += 2
            items = [self._read() for _ in range(count)]
            return '[' + ', '.join(str(i) for i in items) + ']'
        elif t == DICT:
            count = struct.unpack_from('<H', self.code, self.pos)[0]
            self.pos += 2
            items = [f'{self._read()}: {self._read()}' for _ in range(count)]
            return '{' + ', '.join(items) + '}'
        elif t == CALL:
            name = self._read()
            args = []
            while self.pos < len(self.code) and self.code[self.pos] != END:
                args.append(self._read())
            if self.pos < len(self.code):
                self.pos += 1
            return f'{name}({", ".join(str(a) for a in args)})'
        elif t in (ADD, SUB, MUL, DIV, MOD, POW):
            op = {ADD: '+', SUB: '-', MUL: '*', DIV: '/', MOD: '%', POW: '**'}[t]
            return op
        elif t in (EQ, NEQ, LT, GT, LTE, GTE):
            op = {EQ: '==', NEQ: '!=', LT: '<', GT: '>', LTE: '<=', GTE: '>='}[t]
            return op
        elif t in (AND, OR):
            return 'and' if t == AND else 'or'
        elif t == NOT:
            return 'not'
        elif t == NEG:
            return '-'
        return f'<0x{t:02X}>'
