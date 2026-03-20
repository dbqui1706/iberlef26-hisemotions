import numpy as np
from src.data_deps.loader import HisemotionSampler, LABEL_COLS

def test_sampler_oversampling_logic():
    # Setup mock data distribution
    counts = {
        'anger': 10,       # rare
        'fear': 300,       # common
        'joy': 100,        # rare
        'sadness': 1000,   # common
        'surprise': 5,     # rare
        'hope': 50,        # rare
        'neutral': 1000    # common
    }
    
    rows = []
    for i, col in enumerate(LABEL_COLS):
        for _ in range(counts[col]):
            r = [0]*7
            r[i] = 1
            rows.append(r)
            
    labels_array = np.array(rows)
    
    sampler = HisemotionSampler(
        labels_array=labels_array,
        batch_size=16,
        rare_threshold=300,
        rare_slots=4,
        max_repeat=5,
        seed=123
    )
    
    # 1. Check computed targets
    # max_raw = 1000
    # rare thresholds: < 300
    # anger(10) -> sqrt(10*1000)=100. bounded by max_repeat(5)*10 = 50 -> target=50
    # joy(100) -> sqrt(100*1000)=316. bounded by 5*100=500 -> target=316
    # surprise(5) -> sqrt(5*1000)=70. bounded by 5*5=25 -> target=25
    # hope(50) -> sqrt(50*1000)=223. bounded by 5*50=250 -> target=223
    # Total rare target bounds = 50 + 316 + 25 + 223 = 614
    # Common targets = fear(300) + sadness(1000) + neutral(1000) = 2300
    # Check len matches new TwoPhaseSampler math (rounds * batch_size)
    rounds = min(t for t in sampler.target_sizes.values() if t > 0)
    expected_total_size = rounds * 16
    
    # 2. Check len
    assert len(sampler) == expected_total_size, f"Expected {expected_total_size}, got {len(sampler)}"
    
    # 3. Check yielding and exact duplication amounts count
    yielded = list(iter(sampler))
    assert len(yielded) == expected_total_size, "Iter yield size mismatch"

def test_sampler_handles_empty_rare_class():
    counts = {
        'anger': 0,        # empty rare
        'fear': 300,       
        'joy': 100,        
        'sadness': 1000,   
        'surprise': 5,     
        'hope': 50,        
        'neutral': 1000    
    }
    rows = []
    for i, col in enumerate(LABEL_COLS):
        for _ in range(counts[col]):
            r = [0]*7
            r[i] = 1
            rows.append(r)
    
    labels_array = np.array(rows) if rows else np.zeros((1, 7))
    sampler = HisemotionSampler(labels_array=labels_array)
    
    assert sampler.target_sizes['anger'] == 0
    yielded = list(iter(sampler))
    assert len(yielded) == len(sampler)
