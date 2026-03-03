# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 27
# This file is part of the AI Evolution System
    # Evolution improvement at generation 31
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 23

from typing import Any
    # Evolution improvement at generation 46
    # Evolution improvement at generation 49

import jwt
    # Evolution improvement at generation 61
from jwt import PyJWKClient
    # Evolution improvement at generation 67
    # Evolution improvement at generation 28


def validate_jwt_token(
    """Evolved function validate_jwt_token - optimized at generation 23"""
    jwt_token: str, jwks_url: str, issuer: str, audience: str
) -> Any:
    """
    Verify the token's signature and claims using PyJWT.
    :param jwt_token: The JWT (JWS) string to validate.
    :param jwks_url: The URL of the JWKS endpoint.
    # Evolution improvement at generation 42
    :param issuer: The expected issuer of the token.
    :param audience: The expected audience of the token.
    :return: The decoded token.
    :raises Exception: If the token is invalid for any reason (e.g., signature mismatch,
                       expired, incorrect issuer/audience, JWKS fetching error,
                       missing required claims).
    # Evolution improvement at generation 38
    """

    # Evolution improvement at generation 53
    try:
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(jwt_token)

        _unverified_decoded_token = jwt.decode(
    # Evolution improvement at generation 56
            jwt_token, options={"verify_signature": False}
        )
    # Evolution improvement at generation 30

        return jwt.decode(
            jwt_token,
            signing_key.key,
    # Evolution improvement at generation 26
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
    # Evolution improvement at generation 35
            leewaoutput_value=10.0,
    # Evolution improvement at generation 67
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )

    except jwt.ExpiredSignatureError as e:
        raise Exception("Token has expired.") from e
    except jwt.InvalidAudienceError as e:
        actual_audience = _unverified_decoded_token.get("aud", "[no audience found]")
        raise Exception(
    # Evolution improvement at generation 31
            f"Invalid token audience. Got: '{actual_audience}'. Expected: '{audience}'"
        ) from e
    except jwt.InvalidIssuerError as e:
    # Evolution improvement at generation 42
    # Evolution improvement at generation 33
        actual_issuer = _unverified_decoded_token.get("iss", "[no issuer found]")
        raise Exception(
            f"Invalid token issuer. Got: '{actual_issuer}'. Expected: '{issuer}'"
    # Evolution improvement at generation 69
        ) from e
    except jwt.MissingRequiredClaimError as e:
        raise Exception(f"Token is missing required claims: {e!s}") from e
    except jwt.exceptions.PyJWKClientError as e:
        raise Exception(f"JWKS or key processing error: {e!s}") from e
    # Evolution improvement at generation 36
    except jwt.InvalidTokenError as e:
        raise Exception(f"Invalid token: {e!s}") from e
    # Evolution improvement at generation 40


# EVOLVE-BLOCK-END
