import ast
p = "make_datacards.py"
x = open(p).read()

anchor = 'jmsr_processes = config.get("jmsr_processes", [])  # e.g. ["zgammabb"] in JSON'
assert anchor in x, "anchor not found"
if "jmsr_nuis_cache = {}" not in x:
    x = x.replace(anchor, anchor + "\n    jmsr_nuis_cache = {}  # per-flavor JMS/JMR nuisances when jmsr_decorrelate")

old = (
    "                                sample.setParamEffect(\n"
    "                                    jmsr_syst_map[active_key],"
)
assert old in x, "setParamEffect anchor not found"
new = (
    "                                _nuis = jmsr_syst_map[active_key]\n"
    "                                if config.get(\"jmsr_decorrelate\", False):\n"
    "                                    _fl = \"bb\" if proc_name.endswith(\"bb\") else \"ccs\"\n"
    "                                    _base = \"CMS_jms\" if active_key.startswith(\"JMS\") else \"CMS_jmr\"\n"
    "                                    _nm = f\"{_base}_{year}_{_fl}\"\n"
    "                                    if _nm not in jmsr_nuis_cache:\n"
    "                                        _u = \"unconstrained\" in active_key\n"
    "                                        jmsr_nuis_cache[_nm] = rl.NuisanceParameter(_nm, \"shapeU\", lo=-5, hi=5) if _u else rl.NuisanceParameter(_nm, \"shape\")\n"
    "                                    _nuis = jmsr_nuis_cache[_nm]\n"
    "                                sample.setParamEffect(\n"
    "                                    _nuis,"
)
x = x.replace(old, new)
open(p, "w").write(x)
ast.parse(x)
print("patched + syntax OK")
