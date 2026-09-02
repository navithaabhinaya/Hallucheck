"""
Golden Set for Fabricated API Detection (Stage 3, first slice)

Each case is a small code snippet with KNOWN, hand-labeled expected verdicts
for its calls -- this is the ground truth the scorer checks the real
checker's output against.

Deliberately includes:
- A clean case (nothing fabricated) -- checks we don't false-positive
- A fabricated import
- A fabricated method on a real package (the harder, more realistic case)
- A multi-level real attribute chain (os.path.join) -- regression test for
  the bug we just fixed
- An instance-method case we EXPECT to be UNVERIFIED (known limitation,
  not a failure) -- tests that the scorer treats "honestly uncertain"
  correctly, not as a wrong answer
"""

GOLDEN_SET = [
    {
        "id": "case_01_clean",
        "code": '''
import os
def list_files(path):
    return os.listdir(path)
''',
        "expected_calls": {
            "os.listdir": "real",
        },
    },
    {
        "id": "case_02_fabricated_import",
        "code": '''
import totally_fake_package_xyz123
def do_thing():
    return totally_fake_package_xyz123.run()
''',
        "expected_calls": {
            # call is UNVERIFIED because the import itself is fabricated
            # (de-duplication: we don't double-flag the same root cause)
            "totally_fake_package_xyz123.run": "unverified",
        },
    },
    {
        "id": "case_03_fabricated_method_on_real_package",
        "code": '''
import requests
def fetch(url):
    return requests.fetch_url(url)
''',
        "expected_calls": {
            "requests.fetch_url": "fabricated",
        },
    },
    {
        "id": "case_04_multilevel_real_attribute",
        "code": '''
import os
def join_paths(a, b):
    return os.path.join(a, b)
''',
        "expected_calls": {
            "os.path.join": "real",
        },
    },
    {
        "id": "case_05_instance_method_known_limitation",
        "code": '''
import requests
def get_json(url):
    resp = requests.get(url)
    return resp.json()
''',
        "expected_calls": {
            "requests.get": "real",
            # resp.json() -- resp is a variable, not the module itself.
            # We EXPECT unverified here -- this is the documented limitation,
            # not a bug. The scorer should treat this as a correct match,
            # not count it as a miss.
            "resp.json": "unverified",
        },
    },
]
