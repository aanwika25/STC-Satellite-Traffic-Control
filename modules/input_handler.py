def _tle_checksum(line: str) -> int:
    """Modulo-10 checksum over columns 1-68; '-' counts as 1."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def validate_tle(line1: str, line2: str) -> bool:
    line1 = line1.rstrip("\r\n")
    line2 = line2.rstrip("\r\n")
    if not line1.startswith("1 "):
        raise ValueError("TLE line 1 must start with '1 '.")
    if not line2.startswith("2 "):
        raise ValueError("TLE line 2 must start with '2 '.")
    if len(line1) != 69 or len(line2) != 69:
        raise ValueError(
            f"Each TLE line must contain exactly 69 characters "
            f"(received {len(line1)} and {len(line2)})."
        )
    if line1[2:7] != line2[2:7]:
        raise ValueError("The catalogue numbers in the two TLE lines do not match.")
    for label, line in (("line 1", line1), ("line 2", line2)):
        if not line[68].isdigit() or _tle_checksum(line) != int(line[68]):
            raise ValueError(
                f"TLE {label} failed its checksum. Check the record for a copying error."
            )
    return True


def validate_station(latitude, longitude, altitude_km):
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    if not -0.5 <= altitude_km <= 10.0:
        raise ValueError("Ground-station altitude must be between -0.5 and 10 km.")
    return True


def validate_simulation(duration_hours, timestep_seconds, frequency_mhz,
                        minimum_elevation_deg, screening_distance_km):
    if duration_hours <= 0:
        raise ValueError("Simulation duration must be positive.")
    if timestep_seconds <= 0:
        raise ValueError("Time step must be positive.")
    if timestep_seconds > duration_hours * 3600:
        raise ValueError("Time step cannot be longer than the simulation duration.")
    if frequency_mhz <= 0:
        raise ValueError("Carrier frequency must be positive.")
    if not 0 <= minimum_elevation_deg <= 90:
        raise ValueError("Minimum elevation must be between 0 and 90 degrees.")
    if screening_distance_km <= 0:
        raise ValueError("Screening distance must be positive.")
    return True
