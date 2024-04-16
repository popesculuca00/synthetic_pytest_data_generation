def percent_slower(new, old):
    
    assert new >= old
    precent_slower = (old - new) / old * 100
    return precent_slower