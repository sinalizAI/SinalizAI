from typing import Dict, Any, Optional, List
from datetime import datetime


class BenchmarkModel:
    def __init__(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.duration = duration
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'duration': self.duration,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkModel':
        instance = cls(
            operation=data['operation'],
            duration=data['duration'],
            metadata=data.get('metadata', {})
        )
        
        if 'timestamp' in data:
            try:
                instance.timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                pass
        
        return instance
    
    def get_duration_ms(self) -> float:
        return self.duration * 1000
    
    def get_duration_formatted(self) -> str:
        ms = self.get_duration_ms()
        if ms < 1000:
            return f"{ms:.1f}ms"
        else:
            return f"{self.duration:.2f}s"
    
    def is_slow(self, threshold_seconds: float = 2.0) -> bool:
        return self.duration > threshold_seconds
    
    def get_performance_level(self) -> str:
        if self.duration < 0.5:
            return "Excelente"
        elif self.duration < 1.0:
            return "Bom"
        elif self.duration < 2.0:
            return "Aceitável"
        else:
            return "Lento"
    
    def __str__(self) -> str:
        return f"Benchmark({self.operation}: {self.get_duration_formatted()})"
    
    def __repr__(self) -> str:
        return self.__str__()


class BenchmarkCollection:
    def __init__(self):
        self.benchmarks: List[BenchmarkModel] = []
    
    def add_benchmark(self, benchmark: BenchmarkModel):
        self.benchmarks.append(benchmark)
    
    def get_average_duration(self, operation: str = None) -> float:
        filtered_benchmarks = self.benchmarks
        
        if operation:
            filtered_benchmarks = [b for b in self.benchmarks if b.operation == operation]
        
        if not filtered_benchmarks:
            return 0.0
        
        total_duration = sum(b.duration for b in filtered_benchmarks)
        return total_duration / len(filtered_benchmarks)
    
    def get_slowest_operations(self, limit: int = 5) -> List[BenchmarkModel]:
        return sorted(self.benchmarks, key=lambda b: b.duration, reverse=True)[:limit]
    
    def get_operations_summary(self) -> Dict[str, Dict[str, Any]]:
        operations = {}
        
        for benchmark in self.benchmarks:
            op = benchmark.operation
            if op not in operations:
                operations[op] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0
                }
            
            operations[op]['count'] += 1
            operations[op]['total_duration'] += benchmark.duration
            operations[op]['min_duration'] = min(operations[op]['min_duration'], benchmark.duration)
            operations[op]['max_duration'] = max(operations[op]['max_duration'], benchmark.duration)
        
        for op_data in operations.values():
            op_data['avg_duration'] = op_data['total_duration'] / op_data['count']
        
        return operations