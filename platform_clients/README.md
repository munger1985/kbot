# Platform Clients

This package owns HTTP clients used to communicate with model services, the
SQL executor and operations integrations. Clients carry internal
authentication, timeout and response normalization; they do not import
service repositories or ORM entities.
