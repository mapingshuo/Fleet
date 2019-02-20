// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: brpc/nshead_meta.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "brpc/nshead_meta.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace brpc {

namespace {

const ::google::protobuf::Descriptor* NsheadMeta_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  NsheadMeta_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_brpc_2fnshead_5fmeta_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AssignDesc_brpc_2fnshead_5fmeta_2eproto() {
  protobuf_AddDesc_brpc_2fnshead_5fmeta_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "brpc/nshead_meta.proto");
  GOOGLE_CHECK(file != NULL);
  NsheadMeta_descriptor_ = file->message_type(0);
  static const int NsheadMeta_offsets_[9] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, full_method_name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, correlation_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, log_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, attachment_size_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, compress_type_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, trace_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, span_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, parent_span_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, user_string_),
  };
  NsheadMeta_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      NsheadMeta_descriptor_,
      NsheadMeta::default_instance_,
      NsheadMeta_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, _has_bits_[0]),
      -1,
      -1,
      sizeof(NsheadMeta),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NsheadMeta, _internal_metadata_),
      -1);
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_brpc_2fnshead_5fmeta_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      NsheadMeta_descriptor_, &NsheadMeta::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_brpc_2fnshead_5fmeta_2eproto() {
  delete NsheadMeta::default_instance_;
  delete NsheadMeta_reflection_;
}

void protobuf_AddDesc_brpc_2fnshead_5fmeta_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AddDesc_brpc_2fnshead_5fmeta_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::brpc::protobuf_AddDesc_brpc_2foptions_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\026brpc/nshead_meta.proto\022\004brpc\032\022brpc/opt"
    "ions.proto\"\342\001\n\nNsheadMeta\022\030\n\020full_method"
    "_name\030\001 \002(\t\022\026\n\016correlation_id\030\002 \001(\003\022\016\n\006l"
    "og_id\030\003 \001(\003\022\027\n\017attachment_size\030\004 \001(\005\022)\n\r"
    "compress_type\030\005 \001(\0162\022.brpc.CompressType\022"
    "\020\n\010trace_id\030\006 \001(\003\022\017\n\007span_id\030\007 \001(\003\022\026\n\016pa"
    "rent_span_id\030\010 \001(\003\022\023\n\013user_string\030\t \001(\014B"
    "\027\n\010com.brpcB\013NsheadProto", 304);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "brpc/nshead_meta.proto", &protobuf_RegisterTypes);
  NsheadMeta::default_instance_ = new NsheadMeta();
  NsheadMeta::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_brpc_2fnshead_5fmeta_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_brpc_2fnshead_5fmeta_2eproto {
  StaticDescriptorInitializer_brpc_2fnshead_5fmeta_2eproto() {
    protobuf_AddDesc_brpc_2fnshead_5fmeta_2eproto();
  }
} static_descriptor_initializer_brpc_2fnshead_5fmeta_2eproto_;

// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int NsheadMeta::kFullMethodNameFieldNumber;
const int NsheadMeta::kCorrelationIdFieldNumber;
const int NsheadMeta::kLogIdFieldNumber;
const int NsheadMeta::kAttachmentSizeFieldNumber;
const int NsheadMeta::kCompressTypeFieldNumber;
const int NsheadMeta::kTraceIdFieldNumber;
const int NsheadMeta::kSpanIdFieldNumber;
const int NsheadMeta::kParentSpanIdFieldNumber;
const int NsheadMeta::kUserStringFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

NsheadMeta::NsheadMeta()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:brpc.NsheadMeta)
}

void NsheadMeta::InitAsDefaultInstance() {
}

NsheadMeta::NsheadMeta(const NsheadMeta& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:brpc.NsheadMeta)
}

void NsheadMeta::SharedCtor() {
  ::google::protobuf::internal::GetEmptyString();
  _cached_size_ = 0;
  full_method_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  correlation_id_ = GOOGLE_LONGLONG(0);
  log_id_ = GOOGLE_LONGLONG(0);
  attachment_size_ = 0;
  compress_type_ = 0;
  trace_id_ = GOOGLE_LONGLONG(0);
  span_id_ = GOOGLE_LONGLONG(0);
  parent_span_id_ = GOOGLE_LONGLONG(0);
  user_string_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

NsheadMeta::~NsheadMeta() {
  // @@protoc_insertion_point(destructor:brpc.NsheadMeta)
  SharedDtor();
}

void NsheadMeta::SharedDtor() {
  full_method_name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  user_string_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (this != default_instance_) {
  }
}

void NsheadMeta::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* NsheadMeta::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return NsheadMeta_descriptor_;
}

const NsheadMeta& NsheadMeta::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_brpc_2fnshead_5fmeta_2eproto();
  return *default_instance_;
}

NsheadMeta* NsheadMeta::default_instance_ = NULL;

NsheadMeta* NsheadMeta::New(::google::protobuf::Arena* arena) const {
  NsheadMeta* n = new NsheadMeta;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void NsheadMeta::Clear() {
// @@protoc_insertion_point(message_clear_start:brpc.NsheadMeta)
#if defined(__clang__)
#define ZR_HELPER_(f) \
  _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Winvalid-offsetof\"") \
  __builtin_offsetof(NsheadMeta, f) \
  _Pragma("clang diagnostic pop")
#else
#define ZR_HELPER_(f) reinterpret_cast<char*>(\
  &reinterpret_cast<NsheadMeta*>(16)->f)
#endif

#define ZR_(first, last) do {\
  ::memset(&first, 0,\
           ZR_HELPER_(last) - ZR_HELPER_(first) + sizeof(last));\
} while (0)

  if (_has_bits_[0 / 32] & 255u) {
    ZR_(correlation_id_, parent_span_id_);
    if (has_full_method_name()) {
      full_method_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    }
  }
  if (has_user_string()) {
    user_string_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }

#undef ZR_HELPER_
#undef ZR_

  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  if (_internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->Clear();
  }
}

bool NsheadMeta::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:brpc.NsheadMeta)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required string full_method_name = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_full_method_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
            this->full_method_name().data(), this->full_method_name().length(),
            ::google::protobuf::internal::WireFormat::PARSE,
            "brpc.NsheadMeta.full_method_name");
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(16)) goto parse_correlation_id;
        break;
      }

      // optional int64 correlation_id = 2;
      case 2: {
        if (tag == 16) {
         parse_correlation_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &correlation_id_)));
          set_has_correlation_id();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(24)) goto parse_log_id;
        break;
      }

      // optional int64 log_id = 3;
      case 3: {
        if (tag == 24) {
         parse_log_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &log_id_)));
          set_has_log_id();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(32)) goto parse_attachment_size;
        break;
      }

      // optional int32 attachment_size = 4;
      case 4: {
        if (tag == 32) {
         parse_attachment_size:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &attachment_size_)));
          set_has_attachment_size();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(40)) goto parse_compress_type;
        break;
      }

      // optional .brpc.CompressType compress_type = 5;
      case 5: {
        if (tag == 40) {
         parse_compress_type:
          int value;
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   int, ::google::protobuf::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
          if (::brpc::CompressType_IsValid(value)) {
            set_compress_type(static_cast< ::brpc::CompressType >(value));
          } else {
            mutable_unknown_fields()->AddVarint(5, value);
          }
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(48)) goto parse_trace_id;
        break;
      }

      // optional int64 trace_id = 6;
      case 6: {
        if (tag == 48) {
         parse_trace_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &trace_id_)));
          set_has_trace_id();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(56)) goto parse_span_id;
        break;
      }

      // optional int64 span_id = 7;
      case 7: {
        if (tag == 56) {
         parse_span_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &span_id_)));
          set_has_span_id();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(64)) goto parse_parent_span_id;
        break;
      }

      // optional int64 parent_span_id = 8;
      case 8: {
        if (tag == 64) {
         parse_parent_span_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &parent_span_id_)));
          set_has_parent_span_id();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(74)) goto parse_user_string;
        break;
      }

      // optional bytes user_string = 9;
      case 9: {
        if (tag == 74) {
         parse_user_string:
          DO_(::google::protobuf::internal::WireFormatLite::ReadBytes(
                input, this->mutable_user_string()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:brpc.NsheadMeta)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:brpc.NsheadMeta)
  return false;
#undef DO_
}

void NsheadMeta::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:brpc.NsheadMeta)
  // required string full_method_name = 1;
  if (has_full_method_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->full_method_name().data(), this->full_method_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "brpc.NsheadMeta.full_method_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->full_method_name(), output);
  }

  // optional int64 correlation_id = 2;
  if (has_correlation_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(2, this->correlation_id(), output);
  }

  // optional int64 log_id = 3;
  if (has_log_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(3, this->log_id(), output);
  }

  // optional int32 attachment_size = 4;
  if (has_attachment_size()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(4, this->attachment_size(), output);
  }

  // optional .brpc.CompressType compress_type = 5;
  if (has_compress_type()) {
    ::google::protobuf::internal::WireFormatLite::WriteEnum(
      5, this->compress_type(), output);
  }

  // optional int64 trace_id = 6;
  if (has_trace_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(6, this->trace_id(), output);
  }

  // optional int64 span_id = 7;
  if (has_span_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(7, this->span_id(), output);
  }

  // optional int64 parent_span_id = 8;
  if (has_parent_span_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(8, this->parent_span_id(), output);
  }

  // optional bytes user_string = 9;
  if (has_user_string()) {
    ::google::protobuf::internal::WireFormatLite::WriteBytesMaybeAliased(
      9, this->user_string(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:brpc.NsheadMeta)
}

::google::protobuf::uint8* NsheadMeta::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:brpc.NsheadMeta)
  // required string full_method_name = 1;
  if (has_full_method_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->full_method_name().data(), this->full_method_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "brpc.NsheadMeta.full_method_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->full_method_name(), target);
  }

  // optional int64 correlation_id = 2;
  if (has_correlation_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(2, this->correlation_id(), target);
  }

  // optional int64 log_id = 3;
  if (has_log_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(3, this->log_id(), target);
  }

  // optional int32 attachment_size = 4;
  if (has_attachment_size()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(4, this->attachment_size(), target);
  }

  // optional .brpc.CompressType compress_type = 5;
  if (has_compress_type()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteEnumToArray(
      5, this->compress_type(), target);
  }

  // optional int64 trace_id = 6;
  if (has_trace_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(6, this->trace_id(), target);
  }

  // optional int64 span_id = 7;
  if (has_span_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(7, this->span_id(), target);
  }

  // optional int64 parent_span_id = 8;
  if (has_parent_span_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(8, this->parent_span_id(), target);
  }

  // optional bytes user_string = 9;
  if (has_user_string()) {
    target =
      ::google::protobuf::internal::WireFormatLite::WriteBytesToArray(
        9, this->user_string(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:brpc.NsheadMeta)
  return target;
}

int NsheadMeta::ByteSize() const {
// @@protoc_insertion_point(message_byte_size_start:brpc.NsheadMeta)
  int total_size = 0;

  // required string full_method_name = 1;
  if (has_full_method_name()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->full_method_name());
  }
  if (_has_bits_[1 / 32] & 254u) {
    // optional int64 correlation_id = 2;
    if (has_correlation_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int64Size(
          this->correlation_id());
    }

    // optional int64 log_id = 3;
    if (has_log_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int64Size(
          this->log_id());
    }

    // optional int32 attachment_size = 4;
    if (has_attachment_size()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->attachment_size());
    }

    // optional .brpc.CompressType compress_type = 5;
    if (has_compress_type()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::EnumSize(this->compress_type());
    }

    // optional int64 trace_id = 6;
    if (has_trace_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int64Size(
          this->trace_id());
    }

    // optional int64 span_id = 7;
    if (has_span_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int64Size(
          this->span_id());
    }

    // optional int64 parent_span_id = 8;
    if (has_parent_span_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int64Size(
          this->parent_span_id());
    }

  }
  // optional bytes user_string = 9;
  if (has_user_string()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::BytesSize(
        this->user_string());
  }

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void NsheadMeta::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:brpc.NsheadMeta)
  if (GOOGLE_PREDICT_FALSE(&from == this)) {
    ::google::protobuf::internal::MergeFromFail(__FILE__, __LINE__);
  }
  const NsheadMeta* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const NsheadMeta>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:brpc.NsheadMeta)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:brpc.NsheadMeta)
    MergeFrom(*source);
  }
}

void NsheadMeta::MergeFrom(const NsheadMeta& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:brpc.NsheadMeta)
  if (GOOGLE_PREDICT_FALSE(&from == this)) {
    ::google::protobuf::internal::MergeFromFail(__FILE__, __LINE__);
  }
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_full_method_name()) {
      set_has_full_method_name();
      full_method_name_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.full_method_name_);
    }
    if (from.has_correlation_id()) {
      set_correlation_id(from.correlation_id());
    }
    if (from.has_log_id()) {
      set_log_id(from.log_id());
    }
    if (from.has_attachment_size()) {
      set_attachment_size(from.attachment_size());
    }
    if (from.has_compress_type()) {
      set_compress_type(from.compress_type());
    }
    if (from.has_trace_id()) {
      set_trace_id(from.trace_id());
    }
    if (from.has_span_id()) {
      set_span_id(from.span_id());
    }
    if (from.has_parent_span_id()) {
      set_parent_span_id(from.parent_span_id());
    }
  }
  if (from._has_bits_[8 / 32] & (0xffu << (8 % 32))) {
    if (from.has_user_string()) {
      set_has_user_string();
      user_string_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.user_string_);
    }
  }
  if (from._internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->MergeFrom(from.unknown_fields());
  }
}

void NsheadMeta::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:brpc.NsheadMeta)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void NsheadMeta::CopyFrom(const NsheadMeta& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:brpc.NsheadMeta)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool NsheadMeta::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000001) != 0x00000001) return false;

  return true;
}

void NsheadMeta::Swap(NsheadMeta* other) {
  if (other == this) return;
  InternalSwap(other);
}
void NsheadMeta::InternalSwap(NsheadMeta* other) {
  full_method_name_.Swap(&other->full_method_name_);
  std::swap(correlation_id_, other->correlation_id_);
  std::swap(log_id_, other->log_id_);
  std::swap(attachment_size_, other->attachment_size_);
  std::swap(compress_type_, other->compress_type_);
  std::swap(trace_id_, other->trace_id_);
  std::swap(span_id_, other->span_id_);
  std::swap(parent_span_id_, other->parent_span_id_);
  user_string_.Swap(&other->user_string_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata NsheadMeta::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = NsheadMeta_descriptor_;
  metadata.reflection = NsheadMeta_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// NsheadMeta

// required string full_method_name = 1;
bool NsheadMeta::has_full_method_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void NsheadMeta::set_has_full_method_name() {
  _has_bits_[0] |= 0x00000001u;
}
void NsheadMeta::clear_has_full_method_name() {
  _has_bits_[0] &= ~0x00000001u;
}
void NsheadMeta::clear_full_method_name() {
  full_method_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_full_method_name();
}
 const ::std::string& NsheadMeta::full_method_name() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.full_method_name)
  return full_method_name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 void NsheadMeta::set_full_method_name(const ::std::string& value) {
  set_has_full_method_name();
  full_method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.full_method_name)
}
 void NsheadMeta::set_full_method_name(const char* value) {
  set_has_full_method_name();
  full_method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:brpc.NsheadMeta.full_method_name)
}
 void NsheadMeta::set_full_method_name(const char* value, size_t size) {
  set_has_full_method_name();
  full_method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:brpc.NsheadMeta.full_method_name)
}
 ::std::string* NsheadMeta::mutable_full_method_name() {
  set_has_full_method_name();
  // @@protoc_insertion_point(field_mutable:brpc.NsheadMeta.full_method_name)
  return full_method_name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 ::std::string* NsheadMeta::release_full_method_name() {
  // @@protoc_insertion_point(field_release:brpc.NsheadMeta.full_method_name)
  clear_has_full_method_name();
  return full_method_name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 void NsheadMeta::set_allocated_full_method_name(::std::string* full_method_name) {
  if (full_method_name != NULL) {
    set_has_full_method_name();
  } else {
    clear_has_full_method_name();
  }
  full_method_name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), full_method_name);
  // @@protoc_insertion_point(field_set_allocated:brpc.NsheadMeta.full_method_name)
}

// optional int64 correlation_id = 2;
bool NsheadMeta::has_correlation_id() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
void NsheadMeta::set_has_correlation_id() {
  _has_bits_[0] |= 0x00000002u;
}
void NsheadMeta::clear_has_correlation_id() {
  _has_bits_[0] &= ~0x00000002u;
}
void NsheadMeta::clear_correlation_id() {
  correlation_id_ = GOOGLE_LONGLONG(0);
  clear_has_correlation_id();
}
 ::google::protobuf::int64 NsheadMeta::correlation_id() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.correlation_id)
  return correlation_id_;
}
 void NsheadMeta::set_correlation_id(::google::protobuf::int64 value) {
  set_has_correlation_id();
  correlation_id_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.correlation_id)
}

// optional int64 log_id = 3;
bool NsheadMeta::has_log_id() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
void NsheadMeta::set_has_log_id() {
  _has_bits_[0] |= 0x00000004u;
}
void NsheadMeta::clear_has_log_id() {
  _has_bits_[0] &= ~0x00000004u;
}
void NsheadMeta::clear_log_id() {
  log_id_ = GOOGLE_LONGLONG(0);
  clear_has_log_id();
}
 ::google::protobuf::int64 NsheadMeta::log_id() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.log_id)
  return log_id_;
}
 void NsheadMeta::set_log_id(::google::protobuf::int64 value) {
  set_has_log_id();
  log_id_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.log_id)
}

// optional int32 attachment_size = 4;
bool NsheadMeta::has_attachment_size() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
void NsheadMeta::set_has_attachment_size() {
  _has_bits_[0] |= 0x00000008u;
}
void NsheadMeta::clear_has_attachment_size() {
  _has_bits_[0] &= ~0x00000008u;
}
void NsheadMeta::clear_attachment_size() {
  attachment_size_ = 0;
  clear_has_attachment_size();
}
 ::google::protobuf::int32 NsheadMeta::attachment_size() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.attachment_size)
  return attachment_size_;
}
 void NsheadMeta::set_attachment_size(::google::protobuf::int32 value) {
  set_has_attachment_size();
  attachment_size_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.attachment_size)
}

// optional .brpc.CompressType compress_type = 5;
bool NsheadMeta::has_compress_type() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
void NsheadMeta::set_has_compress_type() {
  _has_bits_[0] |= 0x00000010u;
}
void NsheadMeta::clear_has_compress_type() {
  _has_bits_[0] &= ~0x00000010u;
}
void NsheadMeta::clear_compress_type() {
  compress_type_ = 0;
  clear_has_compress_type();
}
 ::brpc::CompressType NsheadMeta::compress_type() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.compress_type)
  return static_cast< ::brpc::CompressType >(compress_type_);
}
 void NsheadMeta::set_compress_type(::brpc::CompressType value) {
  assert(::brpc::CompressType_IsValid(value));
  set_has_compress_type();
  compress_type_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.compress_type)
}

// optional int64 trace_id = 6;
bool NsheadMeta::has_trace_id() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
void NsheadMeta::set_has_trace_id() {
  _has_bits_[0] |= 0x00000020u;
}
void NsheadMeta::clear_has_trace_id() {
  _has_bits_[0] &= ~0x00000020u;
}
void NsheadMeta::clear_trace_id() {
  trace_id_ = GOOGLE_LONGLONG(0);
  clear_has_trace_id();
}
 ::google::protobuf::int64 NsheadMeta::trace_id() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.trace_id)
  return trace_id_;
}
 void NsheadMeta::set_trace_id(::google::protobuf::int64 value) {
  set_has_trace_id();
  trace_id_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.trace_id)
}

// optional int64 span_id = 7;
bool NsheadMeta::has_span_id() const {
  return (_has_bits_[0] & 0x00000040u) != 0;
}
void NsheadMeta::set_has_span_id() {
  _has_bits_[0] |= 0x00000040u;
}
void NsheadMeta::clear_has_span_id() {
  _has_bits_[0] &= ~0x00000040u;
}
void NsheadMeta::clear_span_id() {
  span_id_ = GOOGLE_LONGLONG(0);
  clear_has_span_id();
}
 ::google::protobuf::int64 NsheadMeta::span_id() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.span_id)
  return span_id_;
}
 void NsheadMeta::set_span_id(::google::protobuf::int64 value) {
  set_has_span_id();
  span_id_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.span_id)
}

// optional int64 parent_span_id = 8;
bool NsheadMeta::has_parent_span_id() const {
  return (_has_bits_[0] & 0x00000080u) != 0;
}
void NsheadMeta::set_has_parent_span_id() {
  _has_bits_[0] |= 0x00000080u;
}
void NsheadMeta::clear_has_parent_span_id() {
  _has_bits_[0] &= ~0x00000080u;
}
void NsheadMeta::clear_parent_span_id() {
  parent_span_id_ = GOOGLE_LONGLONG(0);
  clear_has_parent_span_id();
}
 ::google::protobuf::int64 NsheadMeta::parent_span_id() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.parent_span_id)
  return parent_span_id_;
}
 void NsheadMeta::set_parent_span_id(::google::protobuf::int64 value) {
  set_has_parent_span_id();
  parent_span_id_ = value;
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.parent_span_id)
}

// optional bytes user_string = 9;
bool NsheadMeta::has_user_string() const {
  return (_has_bits_[0] & 0x00000100u) != 0;
}
void NsheadMeta::set_has_user_string() {
  _has_bits_[0] |= 0x00000100u;
}
void NsheadMeta::clear_has_user_string() {
  _has_bits_[0] &= ~0x00000100u;
}
void NsheadMeta::clear_user_string() {
  user_string_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_user_string();
}
 const ::std::string& NsheadMeta::user_string() const {
  // @@protoc_insertion_point(field_get:brpc.NsheadMeta.user_string)
  return user_string_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 void NsheadMeta::set_user_string(const ::std::string& value) {
  set_has_user_string();
  user_string_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:brpc.NsheadMeta.user_string)
}
 void NsheadMeta::set_user_string(const char* value) {
  set_has_user_string();
  user_string_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:brpc.NsheadMeta.user_string)
}
 void NsheadMeta::set_user_string(const void* value, size_t size) {
  set_has_user_string();
  user_string_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:brpc.NsheadMeta.user_string)
}
 ::std::string* NsheadMeta::mutable_user_string() {
  set_has_user_string();
  // @@protoc_insertion_point(field_mutable:brpc.NsheadMeta.user_string)
  return user_string_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 ::std::string* NsheadMeta::release_user_string() {
  // @@protoc_insertion_point(field_release:brpc.NsheadMeta.user_string)
  clear_has_user_string();
  return user_string_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 void NsheadMeta::set_allocated_user_string(::std::string* user_string) {
  if (user_string != NULL) {
    set_has_user_string();
  } else {
    clear_has_user_string();
  }
  user_string_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), user_string);
  // @@protoc_insertion_point(field_set_allocated:brpc.NsheadMeta.user_string)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace brpc

// @@protoc_insertion_point(global_scope)
