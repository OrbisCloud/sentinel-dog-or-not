from typing import List
from copy import deepcopy


class Matrix:
	
	points: List[List[float]]
	name: str
	width: int
	height: int

	def __init__(self, _points: List[List[float]], _name: str):
		self.points = deepcopy(_points)
		self.name = _name
		self.width = len(_points)
		self.height = len(_points[0])

	def get(self, i: int, j: int) -> float:
		val: float = deepcopy(self.points[i][j])
		return val
		
	def sum_row(self, i: int) -> float:
		return sum(deepcopy(self.points[i]))
	
	def sum_col(self, i: int) -> float:
		sum: float = 0
		for j in range(0, self.width):
			sum += self.get(j, i)
		return sum
	
	def squash(self):		
		m_points: List[float] = [self.sum_col(i) for i in range(0, self.height)]
		return mat_([m_points], f"squash({self.name})")
		
	def sum(self) -> float:
		total: float = sum([self.sum_row(i) for i in range(0, self.width)])
		return total
	
	def apply(self, func, *args):
		# Make sure that the function only takes one argument, which should be a float.
		m_points: List[List[float]] = []
		for i in range(0, self.width):
			m_points.append([])
			for j in range(0, self.height):
				kargs = tuple(arg.get(i, j) for arg in list(args))
				m_points[i].append(func(self.get(i, j), *kargs))
		return mat_(m_points, f"{func.__name__}({self.name})")
	
	def range(self, start: int, end: int):
		m_points: List[List[float]] = []
		i_points: List[List[float]] = deepcopy(self.points)
		for i in range(0, self.width):
			m_points.append(deepcopy(i_points[i][start:end]))
		return mat_(m_points, f"{self.name}[row][{start}: {end}]")
	
	def add(self, add):
		m_points: List[List[float]] = []
		if is_matrix_(add):
			for i in range(0, self.width):
				m_points.append(deepcopy(self.points[i]) + deepcopy(add.points[i]))
			return mat_(m_points, f"{self.name}")
		raise Exception()

	def row(self, start: int, end: int):
		return mat_(deepcopy(self.points[start:end]), f"{self.name}[{start}: {end}][col]")
	
	def __str__(self) -> str:
		banner: str = f"\n			[ {self.name} ] -> \n"
		result: str = ""
		for row in self.points:
			result += "			["
			for val in row:
				result += '%10f' % val
			result += '%5s' % " ] \n"
		result += f"			> Size: ({self.width}x{self.height})\n"
		banner += '%10s' % result
		return banner
	
	def __mul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			m_points: List[List[float]] = []
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) * other)
			return mat_(m_points, f"({self.name} x {other})")
		
		if is_matrix_(other):
			if self.height != other.height:
				print("[mult] Matrices are not the same size")
				print(f"[mult] ({self.width} x {self.height}) vs ({other.width} x {other.height})")
				print(self)
				print(other)
				raise Exception()
			m_points: List[List[float]] = []
			for i in range(0, self.width):
				m_points.append([])
				for ii in range(0, other.width):
					m_points[i].append(0)
					for j in range(0, other.height):
						m_points[i][ii] += (self.get(i, j) * other.get(ii, j))
			return mat_(m_points, f"({self.name} x {other.name})")
		print("[mult] value of type {type(other)} is incompatible." )
		raise Exception()
	
	def __rmul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			m_points: List[List[float]] = []
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) * other)
			return mat_(m_points, f"({other} x {self.name})")
		
		if is_matrix_(other):
			if self.height != other.height:
				print("[rmult] Matrices are not the same size")
				raise Exception()
			
			m_points: List[List[float]] = []
			for i in range(0, self.width):
				m_points.append([])
				for ii in range(0, other.width):
					m_points[i].append(0)
					for j in range(0, other.height):
						m_points[i][ii] += (self.get(i, j) * other.get(ii, j))
			return mat_(m_points, f"({other.name} x {self.name})")
		print("[rmult] value of type {type(other)} is incompatible." )
		raise Exception()
				
	def __add__(self, other):
		m_points: List[List[float]] = []
		if isinstance(other, float) or isinstance(other, int):
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) + other)
			return mat_(m_points, f"({self.name} + {other})")
			
		if is_matrix_(other):
			if self.width != other.width or self.height != other.height:
				print("[add] Matrices are not the same size")
				print(f"[add] ({self.width} x {self.height}) vs ({other.width} x {other.height})")
				print(self)
				print(other)
				raise Exception()
					
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) + other.get(i, j))
			return mat_(m_points, f"({self.name} + {other.name})")
		print("[add] value of type {type(other)} is incompatible." )
		raise Exception()
			
	def __sub__(self, other):
		m_points: List[List[float]] = []
		if isinstance(other, float) or isinstance(other, int):
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) - other)
			return mat_(m_points, f"({self.name} - {other})")
			
		if is_matrix_(other):
			if self.width != other.width or self.height != other.height:
				print("[sub] Matrices are not the same size")
				raise Exception()
			
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) - other.get(i, j))
			return mat_(m_points, f"({self.name} - {other.name})")
		print("[sub] value of type {type(other)} is incompatible." )
		raise Exception()
		
	def __radd__(self, other):
		m_points: List[List[float]] = []
		if isinstance(other, float) or isinstance(other, int):
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) + other)
			return mat_(m_points, f"({self.name} + {other})")
		print("[radd] value of type {type(other)} is incompatible." )
		raise Exception()
	
	def __rsub__(self, other):
		m_points: List[List[float]] = []
		if isinstance(other, float) or isinstance(other, int):
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(other - self.get(i, j))
			return mat_(m_points, f"({other} - {self.name})")
		print("[rsub] value of type {type(other)} is incompatible." )
		raise Exception()
	
	def __pow__(self, other):
		# This operator is placeholder for in-place multiplication
		# Power operator remains unavailable for now
		if is_matrix_(other):
			if self.width != other.width or self.height != other.height:
				print("[.*] Matrices are not the same size")
				print(f"[.*] ({self.width} x {self.height}) vs ({other.width} x {other.height})")
				print(self)
				print(other)
				raise Exception()
				
			m_points: List[List[point]] = []
			for i in range(0, self.width):
				m_points.append([])
				for j in range(0, self.height):
					m_points[i].append(self.get(i, j) * other.get(i, j))
			return mat_(m_points, f"({self.name} ** {other.name})")
		print("[.*] value of type {type(other)} is incompatible." )
		raise Exception()
				

def is_matrix_(object) -> bool:
	return isinstance(object, Matrix)


def mat_(points: List[List[float]], name="matrix", m: Matrix = None) -> Matrix:
	if m:
		return Matrix(deepcopy(m.points), m.name)
	return Matrix(deepcopy(points), name)
	

def vectorize_(main: Matrix, i: int) -> Matrix:
		m_points: List[float] = [row[i] for row in main.points]
		return Matrix([m_points], f"{main.name}(i)({i})")


def row_(main: Matrix, i: int) -> Matrix:
	return Matrix([main.points[i]], f"{main.name}({i})(j)")


def transpose_(main: Matrix) -> Matrix:
	m_points: List[List[float]] = []
	for j in range(0, main.height):
		m_points.append([])
		for i in range(0, main.width):
			m_points[j].append(main.get(i, j))
	return Matrix(m_points, f"transpose({main.name})")


def zero_matrix_(width: int, height: int) -> Matrix:
	m_points: List[List[float]] = []
	for i in range(0, width):
		m_points.append([])
		for j in range(0, height):
			m_points[i].append(0)
	return mat_(m_points, "empty")
	
	
	

