def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def alarm():
    import winsound
    duration = 4000  # millisecond
    freq = 440  # Hz
    winsound.Beep(freq, duration)