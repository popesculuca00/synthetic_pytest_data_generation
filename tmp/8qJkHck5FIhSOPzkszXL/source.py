def derivatives(t, y, vaccine_rate, birth_rate=0.01):
    
    infection_rate = 0.3
    recovery_rate = 0.02
    death_rate = 0.01
    S, I, R = y
    N = S + I + R
    dSdt = (
        -((infection_rate * S * I) / N)
        + ((1 - vaccine_rate) * birth_rate * N)
        - (death_rate * S)
    )
    dIdt = (
        ((infection_rate * S * I) / N)
        - (recovery_rate * I)
        - (death_rate * I)
    )
    dRdt = (
        (recovery_rate * I)
        - (death_rate * R)
        + (vaccine_rate * birth_rate * N)
    )
    return dSdt, dIdt, dRdt