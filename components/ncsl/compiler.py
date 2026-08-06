"""NCSL Compiler — Python AST → NCSL binary bytecode."""
import ast, struct
from .token_table import *

class NCSLCompiler:
    def __init__(self):
        self.out = bytearray()
    
    def compile(self, source: str) -> bytes:
        tree = ast.parse(source)
        self.out = bytearray()
        self.out.extend(MAGIC)
        self.out.append(VERSION)
        self.out.append(0x00)
        self.out.extend(b'\x00\x00\x00\x00')
        self._visit(tree)
        self._emit(END)
        size = len(self.out)
        self.out[4:8] = struct.pack('<I', size)
        return bytes(self.out)
    
    def _emit(self, *bs):
        for b in bs:
            if isinstance(b, int):
                self.out.append(b & 0xFF)
            else:
                self.out.extend(b)
    
    def _emit_str(self, s: str):
        """Emit a string with proper length prefix."""
        data = s.encode('utf-8')
        self._emit(STR)
        if len(data) < 256:
            self.out.append(len(data))
        else:
            self.out.append(0)
            self.out.extend(struct.pack('<H', len(data)))
        self.out.extend(data)
    
    def _emit_int(self, v: int):
        """Emit an integer in compact form."""
        if -128 <= v <= 127:
            self._emit(I8); self.out.extend(struct.pack('<b', v))
        elif -32768 <= v <= 32767:
            self._emit(I16); self.out.extend(struct.pack('<h', v))
        else:
            self._emit(I32); self.out.extend(struct.pack('<i', v))
    
    def _visit(self, node):
        if isinstance(node, ast.Module):
            for s in node.body:
                self._visit(s)
        
        elif isinstance(node, ast.FunctionDef):
            self._emit(FUNC_DEF)
            self._emit_str(node.name)
            for a in node.args.args:
                self._emit_str(a.arg)
            self._emit(END)  # End args
            for s in node.body:
                self._visit(s)
            self._emit(END)  # End function
        
        elif isinstance(node, ast.Return):
            self._emit(RETURN)
            if node.value:
                self._visit(node.value)
        
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._emit_str(t.id)
            self._visit(node.value)
        
        elif isinstance(node, ast.Expr):
            self._visit(node.value)
        
        elif isinstance(node, ast.Call):
            self._emit(CALL)
            if isinstance(node.func, ast.Name):
                self._emit_str(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self._emit_str(node.func.attr)
            for a in node.args:
                self._visit(a)
            self._emit(END)  # End call
        
        elif isinstance(node, ast.Constant):
            if node.value is None:
                self._emit(NONE)
            elif isinstance(node.value, bool):
                self._emit(BOOL, 1 if node.value else 0)
            elif isinstance(node.value, int):
                self._emit_int(node.value)
            elif isinstance(node.value, float):
                self._emit(F64)
                self.out.extend(struct.pack('<d', node.value))
            elif isinstance(node.value, str):
                self._emit_str(node.value)
        
        elif isinstance(node, ast.Name):
            self._emit_str(node.id)
        
        elif isinstance(node, ast.BinOp):
            self._visit(node.left)
            self._visit(node.right)
            m = {ast.Add: ADD, ast.Sub: SUB, ast.Mult: MUL, ast.Div: DIV, ast.Mod: MOD, ast.Pow: POW}
            self._emit(m.get(type(node.op), ADD))
        
        elif isinstance(node, ast.If):
            self._emit(IF)
            self._visit(node.test)
            for s in node.body:
                self._visit(s)
            if node.orelse:
                self._emit(ELSE)
                for s in node.orelse:
                    self._visit(s)
            self._emit(END)
        
        elif isinstance(node, ast.Compare):
            self._visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                self._visit(comp)
                m = {ast.Eq: EQ, ast.NotEq: NEQ, ast.Lt: LT, ast.Gt: GT, ast.LtE: LTE, ast.GtE: GTE}
                self._emit(m.get(type(op), EQ))
        
        elif isinstance(node, ast.BoolOp):
            self._visit(node.values[0])
            if len(node.values) > 1:
                self._visit(node.values[1])
            self._emit(AND if isinstance(node.op, ast.And) else OR)
        
        elif isinstance(node, ast.UnaryOp):
            self._visit(node.operand)
            self._emit(NEG if isinstance(node.op, ast.USub) else NOT)
        
        elif isinstance(node, ast.For):
            self._emit(FOR)
            if isinstance(node.target, ast.Name):
                self._emit_str(node.target.id)
            self._visit(node.iter)
            for s in node.body:
                self._visit(s)
            self._emit(END)
        
        elif isinstance(node, ast.While):
            self._emit(WHILE)
            self._visit(node.test)
            for s in node.body:
                self._visit(s)
            self._emit(END)
        
        elif isinstance(node, ast.Pass):
            self._emit(NOP)
        
        elif isinstance(node, ast.Break):
            self._emit(BREAK)
        
        elif isinstance(node, ast.Continue):
            self._emit(CONTINUE)
        
        elif isinstance(node, ast.List):
            self._emit(LIST)
            self.out.extend(struct.pack('<H', len(node.elts)))
            for e in node.elts:
                if isinstance(e, ast.Constant):
                    self._visit(e)
        
        elif isinstance(node, ast.Dict):
            self._emit(DICT)
            self.out.extend(struct.pack('<H', len(node.keys)))
            for k, v in zip(node.keys, node.values):
                self._visit(k)
                self._visit(v)
