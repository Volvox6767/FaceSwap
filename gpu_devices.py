"""Read DXGI adapter indices used by DirectML; prefer dedicated GPU memory."""
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def dxgi_adapters():
    if os.name != 'nt':
        return []
    import ctypes as c
    from ctypes import wintypes as w
    import uuid

    class GUID(c.Structure):
        _fields_ = [('Data1', w.DWORD), ('Data2', w.WORD), ('Data3', w.WORD), ('Data4', c.c_ubyte * 8)]

    class LUID(c.Structure):
        _fields_ = [('LowPart', w.DWORD), ('HighPart', w.LONG)]

    class DESC(c.Structure):
        _fields_ = [('Description', w.WCHAR * 128), ('VendorId', w.UINT), ('DeviceId', w.UINT),
                    ('SubSysId', w.UINT), ('Revision', w.UINT), ('DedicatedVideoMemory', c.c_size_t),
                    ('DedicatedSystemMemory', c.c_size_t), ('SharedSystemMemory', c.c_size_t),
                    ('AdapterLuid', LUID), ('Flags', w.UINT)]

    def method(pointer, index, restype, args):
        table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(restype, c.c_void_p, *args)(table[index])

    factory = c.c_void_p()
    adapters = []
    try:
        guid = GUID.from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        api = c.WinDLL('dxgi')
        api.CreateDXGIFactory1.argtypes = [c.POINTER(GUID), c.POINTER(c.c_void_p)]
        api.CreateDXGIFactory1.restype = w.LONG
        if api.CreateDXGIFactory1(c.byref(guid), c.byref(factory)) != 0:
            return []
        index = 0
        while True:
            adapter = c.c_void_p()
            status = method(factory, 12, w.LONG, [w.UINT, c.POINTER(c.c_void_p)])(factory, index, c.byref(adapter))
            if status != 0:
                break
            try:
                desc = DESC()
                if method(adapter, 10, w.LONG, [c.POINTER(DESC)])(adapter, c.byref(desc)) == 0:
                    adapters.append({'device_id': index, 'name': desc.Description,
                                     'vram_mb': desc.DedicatedVideoMemory // (1024 ** 2),
                                     'software': bool(desc.Flags & 2)})
            finally:
                method(adapter, 2, w.ULONG, [])(adapter)
            index += 1
    except (OSError, AttributeError):
        return []
    finally:
        if factory:
            method(factory, 2, w.ULONG, [])(factory)
    return adapters


def preferred_dml_device():
    adapters = dxgi_adapters()
    override = os.environ.get('FACESWAP_DML_DEVICE_ID')
    if override is not None:
        index = int(override)
        if index < 0 or (adapters and index not in [a['device_id'] for a in adapters if not a['software']]):
            raise ValueError(f'Invalid DirectML device_id: {index}')
        return index
    hardware = [a for a in adapters if not a['software']]
    return max(hardware, key=lambda a: a['vram_mb'])['device_id'] if hardware else 0
