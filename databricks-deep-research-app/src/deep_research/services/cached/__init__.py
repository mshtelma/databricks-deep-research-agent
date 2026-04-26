"""Cache-backed service implementations.

Each module here implements one service's `I*` Protocol by routing through
`StorageStack` instead of touching SQLAlchemy. Consumers pick the impl via
`services._impl_factory.make_<service>_service(settings, stack, session=…)`
based on `settings.storage_service_impl`.

This package is empty-ish today; Wave 5 lands one service at a time.
"""
