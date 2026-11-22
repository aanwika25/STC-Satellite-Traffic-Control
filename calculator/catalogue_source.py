"""Phase 1 source: pick Object A and Object B from the catalogue loaded in the dashboard sidebar.

Object A is chosen from the satellites (payloads); Object B from the debris and rocket bodies, or from
every object. Each choice is converted into a python-sgp4 Satrec built from the object's own SGP4
mean elements, so Phases 2-5 work on it unchanged.
"""
from __future__ import annotations

import math

import streamlit as st
from sgp4.api import WGS72, Satrec

JD_1949_DEC31 = 2433281.5
DEBRIS_TYPES = ("DEBRIS", "ROCKET BODY")


def satrec_from_catalogue(obj) -> Satrec:
    """python-sgp4 Satrec with the same mean elements as a dashboard object (TLE or OMM)."""
    sat = Satrec()
    # python-sgp4 only accepts catalogue numbers up to 339999 (Alpha-5 "Z9999"). The number is only a
    # label for SGP4, so larger ones (e.g. the synthetic demo debris, 880001+) are passed as 0; the real
    # number is kept in the calculator's "catalogue_number".
    satnum = int(obj.satnum) if str(obj.satnum).isdigit() else 0
    if not 0 <= satnum <= 339999:
        satnum = 0
    sat.sgp4init(WGS72, "i", satnum, obj.jdepoch - JD_1949_DEC31, obj.bstar, 0.0, 0.0,
                 obj.ecco, obj.argpo, obj.inclo, obj.mo, obj.no_kozai, obj.nodeo)
    if sat.error != 0:
        raise ValueError(f"SGP4 could not build {obj.name} (error code {sat.error}).")
    return sat


def _label(o) -> str:
    kind = "debris" if o.obj_type == "DEBRIS" else "rocket body" if o.obj_type == "ROCKET BODY" else \
        ("Indian satellite" if o.country == "IN" else "satellite")
    return f"{o.name} · {kind} · {o.regime} · NORAD {o.satnum}" + (" · synthetic demo" if getattr(o, "synthetic", False) else "")


def _on_pick_a():
    from calculator.link import get_pair, set_pair
    set_pair(st.session_state.get("cat_a"), get_pair().get("b"))


def _on_pick_b():
    from calculator.link import get_pair, set_pair
    set_pair(get_pair().get("a"), st.session_state.get("cat_b"))


def _on_kind(debris_ids):
    """Switched to the debris list while B is a satellite: B becomes the satellite's nearest debris
    approach (from the dashboard screening), or the first debris object."""
    from calculator.link import get_pair, set_pair
    if not st.session_state.get("cat_b_kind", "").startswith("Debris") or get_pair().get("b") in debris_ids:
        return
    ids = set(debris_ids)
    near = [b for b in st.session_state.get("_event_obj_ids", []) if b in ids]
    if near or debris_ids:
        set_pair(get_pair().get("a"), near[0] if near else debris_ids[0])


def catalogue_objects(station_and_radio_inputs, settings):
    """Widgets for the 'Dashboard catalogue' source. Returns the config dict (or None)."""
    data = st.session_state.get("data")
    if not data:
        st.info("No catalogue is loaded in the dashboard.")
        return None
    prims, objs = data["prims"], data["objs"]
    everything = prims + objs
    by_id = {o.satnum: o for o in everything}
    sats = prims + sorted((o for o in objs if o.obj_type not in DEBRIS_TYPES), key=lambda o: o.name)
    debris = sorted((o for o in objs if o.obj_type in DEBRIS_TYPES), key=lambda o: o.name)
    sat_ids = [o.satnum for o in sats]
    st.caption(f"Loaded in the sidebar: **{data['label']}** · {len(sats):,} satellites, {len(debris):,} debris and rocket bodies. "
               "Type in a list to search by name or NORAD number.")

    from calculator.link import get_pair, set_pair
    pair = get_pair()
    st.caption("🔗 **Linked to the dashboard.** A and B follow the satellite in the sidebar and the event chosen on "
               "Collision Stats; changing them here changes the satellite and close approach on every page "
               "(Path Tracer, Collision Stats, Alternate Path Mapping, New Path Stats).")
    if pair.get("a") in sat_ids:
        st.session_state["cat_a"] = pair["a"]
    elif st.session_state.get("cat_a") not in sat_ids:
        st.session_state["cat_a"] = sat_ids[0] if sat_ids else None
    b_obj = by_id.get(pair.get("b"))
    if b_obj is not None:
        if b_obj.obj_type not in DEBRIS_TYPES:
            st.session_state["cat_b_kind"] = "Any object"
        st.session_state["cat_b"] = b_obj.satnum
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Object A** · satellite")
        a_id = st.selectbox("Object A", sat_ids, key="cat_a", format_func=lambda i: _label(by_id[i]), label_visibility="collapsed",
                            on_change=_on_pick_a)
    with c2:
        st.markdown("**Object B**")
        kind = st.radio("Object B list", ["Debris and rocket bodies", "Any object"], horizontal=True, key="cat_b_kind",
                        label_visibility="collapsed", on_change=_on_kind, args=([o.satnum for o in debris],))
        pool = debris if kind == "Debris and rocket bodies" else [o for o in everything if o.satnum != a_id]
        if kind == "Debris and rocket bodies" and not pool:
            st.warning("The loaded data has no debris or rocket bodies (the Space-Track files contain satellites only). "
                       "Showing every object instead. Add Space-Track debris files, or load the sample dataset, for debris.")
            pool = [o for o in everything if o.satnum != a_id]
        b_ids = [o.satnum for o in pool]
        if st.session_state.get("cat_b") not in b_ids:
            st.session_state["cat_b"] = b_ids[0] if b_ids else None
        b_id = st.selectbox("Object B", b_ids, key="cat_b", format_func=lambda i: _label(by_id[i]), label_visibility="collapsed",
                            on_change=_on_pick_b)
    if a_id is None or b_id is None:
        return None
    if a_id == b_id:
        st.warning("Choose two different objects.")
        return None
    if (a_id, b_id) != (pair.get("a"), pair.get("b")):   # e.g. the Object B list was switched
        set_pair(a_id, b_id)
        st.rerun()
    objects = {}
    for key, oid in (("A", a_id), ("B", b_id)):
        o = by_id[oid]
        objects[key] = {"name": o.name, "catalogue_number": o.satnum, "satrec": satrec_from_catalogue(o), "line1": o.line1, "line2": o.line2}
    st.caption("Phases 2 to 5 measure time from Object A's epoch "
               f"({by_id[a_id].name}: TLE/OMM epoch shown in Phase 2).")
    cur = st.session_state.get("cfg")
    if cur and cur.get("linked"):           # keep values the user (or the sidebar) set
        settings = {**settings, "ground_station": {**settings["ground_station"], **cur["ground_station"]},
                    "radio": {**settings["radio"], **cur["radio"]}, "simulation": {**settings["simulation"], **cur["simulation"]}}
    focus = st.session_state.get("calc_focus")
    if focus:
        settings = {**settings, "simulation": {**settings["simulation"], "duration_hours": focus["duration_h"]}}
    settings_in = station_and_radio_inputs(settings, key="cat")
    return {"objects": objects, **settings_in, "mode": "Automated", "linked": True}
