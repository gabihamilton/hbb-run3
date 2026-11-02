import urllib.request, json, ssl

# Ignore SSL errors on LPC
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://cms-gen.cern.ch/xsdb/api/search"

# Just add any other primary dataset names you need to this list!
datasets = [
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYGto2LG-1Jets_Bin-MLL-50-PTG-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
]

for ds in datasets:
    data = json.dumps({"process_name": ds}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, context=ctx).read())
        if resp:
            print(f'"{ds}": {resp[0].get("cross_section")},')
        else:
            print(f"# Not found in XSDB: {ds}")
    except Exception as e:
        print(f"# Error: {e}")
