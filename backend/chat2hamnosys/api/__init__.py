"""HTTP surface for chat2hamnosys.

Public entry point is :func:`create_app` — builds a :class:`FastAPI`
sub-app with the full authoring API (POST /sessions, POST /describe,
POST /answer, POST /correct, POST /accept, GET /events, …). Mount the
returned app under any prefix in the host process::

    from chat2hamnosys.api import create_app

    sub_app = create_app()
    main_app.mount("/api/chat2hamnosys", sub_app)

Tests and ad-hoc scripts can import :data:`router` directly to attach
it to a custom app.
"""

from .app import CORS_ENV_VAR, create_app
from .dependencies import (
    DEFAULT_DATA_DIR,
    DEFAULT_SESSION_DB,
    DEFAULT_SIGN_DB,
    DEFAULT_TOKEN_DB,
    get_apply_fn,
    get_generate_fn,
    get_parse_fn,
    get_question_fn,
    get_render_fn,
    get_session_store,
    get_sign_store,
    get_to_sigml_fn,
    get_token_store,
    reset_stores,
)
from .errors import (
    ApiError,
    ErrorDetail,
    ErrorResponse,
    InvalidTransition,
    SessionForbidden,
    SessionNotFound,
    register_error_handlers,
)
from .models import (
    AcceptResponse,
    AnswerRequest,
    CorrectRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    DescribeRequest,
    GapOut,
    NextAction,
    OptionOut,
    PreviewOut,
    QuestionOut,
    RejectRequest,
    SessionEnvelope,
    SignEntryOut,
)
from .proposals import (
    LanguageProposal,
    LanguageProposalIn,
    LanguageProposalOut,
    ProposalsStore,
    get_proposals_store,
    reset_proposals_store,
)
from .router import DEFAULT_RATE_LIMIT, limiter, router
from .token_store import TokenStore


__all__ = [
    "AcceptResponse",
    "AnswerRequest",
    "ApiError",
    "CORS_ENV_VAR",
    "CorrectRequest",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "DEFAULT_DATA_DIR",
    "DEFAULT_RATE_LIMIT",
    "DEFAULT_SESSION_DB",
    "DEFAULT_SIGN_DB",
    "DEFAULT_TOKEN_DB",
    "DescribeRequest",
    "ErrorDetail",
    "ErrorResponse",
    "GapOut",
    "InvalidTransition",
    "LanguageProposal",
    "LanguageProposalIn",
    "LanguageProposalOut",
    "NextAction",
    "OptionOut",
    "PreviewOut",
    "ProposalsStore",
    "QuestionOut",
    "RejectRequest",
    "SessionEnvelope",
    "SessionForbidden",
    "SessionNotFound",
    "SignEntryOut",
    "TokenStore",
    "create_app",
    "get_apply_fn",
    "get_generate_fn",
    "get_parse_fn",
    "get_proposals_store",
    "get_question_fn",
    "get_render_fn",
    "get_session_store",
    "get_sign_store",
    "get_to_sigml_fn",
    "get_token_store",
    "limiter",
    "register_error_handlers",
    "reset_proposals_store",
    "reset_stores",
    "router",
]
