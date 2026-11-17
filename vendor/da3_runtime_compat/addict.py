"""Subset of addict.Dict needed by upstream DA3, namespaced to isolated runtime.

Do not use to process arbitrary untrusted model objects.
"""
class Dict(dict):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        if name in self:
            return self[name]
        result = type(self)()
        self[name] = result
        return result

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
